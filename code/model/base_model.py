import os
import tensorflow as tf


class BaseModel(object):

    def __init__(self, args):
        """
        初始化函数

        该函数在对象创建时调用，用于设置初始参数和状态。

        参数:
        - args: 包含各种配置或参数的字典或对象，用于初始化实例。

        返回值:
        无

        """
        # 将传入的参数保存到实例变量中，以便其他方法使用
        self.args = args
        # 初始化会话对象为None，它将在以后被赋值
        self.sess = None
        # 初始化编码器的保存器为None，将在训练或需要时被赋值
        self.ed_saver = None
        # 初始化标签器的保存器为None，将在训练或需要时被赋值
        self.el_saver = None

    def reinitialize_weights(self, scope_name):
        """
        重新初始化给定层的权重。

        参数:
        scope_name: str，指定层的作用域名称。用于定位和识别需要重新初始化权重的变量。

        返回:
        无。该方法直接在TensorFlow会话中重新初始化指定层的权重，不返回任何值。
        """
        # 获取指定作用域下的所有变量
        variables = tf.contrib.framework.get_variables(scope_name)
        # 创建一个操作，用于初始化上述变量
        init = tf.variables_initializer(variables)
        # 在当前会话中运行该初始化操作
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """定义self.train_op以在一批数据上执行更新操作
        参数:
            lr_method: (字符串) 优化器方法，例如"adam"
            lr: (tf.placeholder) 学习率的占位符
            loss: (张量) 需要最小化的损失值
            clip: (Python浮点数) 梯度裁剪阈值。如果小于0，不进行裁剪
        """
        _lr_m = lr_method.lower()  # 将方法名转为小写以确保一致性

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # 使用Adam优化器
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':  # 使用Adagrad优化器
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':  # 使用随机梯度下降优化器
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':  # 使用RMSProp优化器
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))  # 如果优化器方法未知，则抛出异常

            if clip > 0:  # 如果clip大于0，则执行梯度裁剪
                grads, vs     = zip(*optimizer.compute_gradients(loss))  # 计算梯度
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)  # 对梯度进行裁剪
                self.train_op = optimizer.apply_gradients(zip(grads, vs))  # 应用裁剪后的梯度
            else:
                self.train_op = optimizer.minimize(loss)  # 直接最小化损失，不进行梯度裁剪

    def initialize_session(self):
        """定义self.sess并初始化变量"""
        print("Initializing tf session")
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)  # 配置会话以使用GPU
        self.sess = tf.Session()  # 创建TensorFlow会话
        # from tensorflow.python import debug as tf_debug
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())  # 启用调试模式
        self.sess.run(tf.global_variables_initializer())  # 初始化所有全局变量
        # 保存特定变量到检查点
        self.ed_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)
        self.el_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)

    def restore_session(self, option="latest"):
        """
        根据指定的选项恢复会话。选项包括'latest'（最新）、'ed'和'el'，其中'latest'会选择ed和el中的最新检查点，
        而'ed'和'el'分别仅选择对应的最新检查点。

        参数:
        - option: str，可选值为"latest"、"ed"、"el"，用于指定恢复哪个检查点。

        返回:
        - checkpoint_path: str，被恢复检查点的路径。
        """
        # 确保option参数是预定义的合法值之一
        assert(option in ["latest", "ed", "el"])

        # 如果指定了具体的模型编号，则根据option选择对应的检查点路径
        if hasattr(self.args, 'checkpoint_model_num') and self.args.checkpoint_model_num is not None:
            assert(option != "latest")  # 如果指定了模型编号，option不能是"latest"
            checkpoint_path = self.args.checkpoints_folder + option + "/model-{}".format(self.args.checkpoint_model_num)
        else:
            # 根据option参数选择检查点路径
            if option == "ed":
                checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"ed/")
            elif option == "el":
                checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"el/")
            elif option == "latest":
                # 当option为"latest"时，选择ed和el中评估次数最多的检查点
                print("Reloading the latest trained model...(either ed or el)")
                ed = self.my_latest_checkpoint(self.args.checkpoints_folder+"ed/")
                el = self.my_latest_checkpoint(self.args.checkpoints_folder+"el/")
                ed_eval_cnt = int(ed[ed.rfind('-') + 1:])
                el_eval_cnt = int(el[el.rfind('-') + 1:])
                if ed_eval_cnt >= el_eval_cnt:
                    checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"ed/")
                    option = "ed"
                else:
                    checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder+"el/")
                    option = "el"

        # 输出使用的检查点路径
        print("Using checkpoint: {}".format(checkpoint_path))

        # 初始化TensorFlow会话和Saver对象
        self.sess = tf.Session()
        self.ed_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)
        self.el_saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.checkpoints_num)

        # 根据option选择对应的Saver对象来恢复模型
        saver = self.ed_saver if option == "ed" else self.el_saver
        saver.restore(self.sess, checkpoint_path)

        # 初始化嵌入层参数
        self.init_embeddings()

        # 完成检查点加载
        print("Finished loading checkpoint.")
        
        # 返回检查点路径
        return checkpoint_path

    def my_latest_checkpoint(self, folder_path):   # model-9.meta
        files = [name for name in os.listdir(folder_path) if name.startswith("model") and name.endswith("meta")]
        max_epoch = max([int(name[len("model-"):-len(".meta")]) for name in files])
        return folder_path + "model-" + str(max_epoch)

    def save_session(self, eval_cnt, save_ed_flag, save_el_flag):
        """
        保存会话 = 权重
        参数:
        - eval_cnt: 评估次数，用于保存时的全局步数
        - save_ed_flag: 是否保存编码器权重的标志
        - save_el_flag: 是否保存定位器权重的标志
        """
        # 遍历保存标志和类别，分别处理编码器和定位器的权重保存
        for save_flag, category in zip([save_ed_flag, save_el_flag], ["ed", "el"]):
            # 如果当前类别的保存标志为False，则跳过
            if save_flag is False:
                continue
            # 设置检查点保存的文件夹路径
            checkpoints_folder = self.args.checkpoints_folder + category + "/"
            # 如果文件夹不存在，则创建
            if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)
            # 打印正在保存的会话检查点类别
            print("saving session checkpoint for {}...".format(category))
            # 设置检查点文件的前缀
            checkpoint_prefix = os.path.join(checkpoints_folder, "model")
            # 根据类别选择相应的saver
            saver = self.ed_saver if category == "ed" else self.el_saver
            # 调用saver保存会话，并打印保存路径
            save_path = saver.save(self.sess, checkpoint_prefix, global_step=eval_cnt)
            print("Checkpoint saved in file: %s" % save_path)

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def _restore_list(self):
        return [n for n in tf.global_variables()
                 if n.name != 'entities/_entity_embeddings:0']

    def checkpoint_variables(self):
        # 决定在保存模型时哪些变量需要排除
        # 当word embeddings和entity embeddings在训练中被固定时，不需要将它们保存在checkpoint中，以节省磁盘空间
        # word embeddings总是被固定，而entity embeddings则根据args.train_ent_vecs的值决定是否固定
        omit_variables = ['words/_word_embeddings:0']  # 总是排除word embeddings
        if not self.args.train_ent_vecs:  # 如果实体向量不参与训练，则也排除entity embeddings
            omit_variables.append('entities/_entity_embeddings:0')
        # 从所有全局变量中筛选出需要保存的变量
        variables = [n for n in tf.global_variables() if n.name not in omit_variables]
        print("checkpoint variables to restore:", variables)
        return variables

    def find_variable_handler_by_name(self, var_name):
        for n in tf.global_variables():
            if n.name == var_name:
                return n