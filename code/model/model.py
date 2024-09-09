
# b9d87f7  on Mar 21 Nikolaos Kolitsas ffnn dropout and some minor modif in evaluate to accept entity extension
# ed_model_21_march
import numpy as np
import pickle
import tensorflow as tf
import model.config as config
from .base_model import BaseModel
import model.util as util


class Model(BaseModel):

    def __init__(self, args, next_element):
        super().__init__(args)
        self.chunk_id, self.words, self.words_len, self.chars, self.chars_len,\
        self.begin_span, self.end_span, self.spans_len,\
        self.cand_entities, self.cand_entities_scores, self.cand_entities_labels,\
        self.cand_entities_len, self.ground_truth, self.ground_truth_len,\
        self.begin_gm, self.end_gm = next_element

        self.begin_span = tf.cast(self.begin_span, tf.int32)
        self.end_span = tf.cast(self.end_span, tf.int32)
        self.words_len = tf.cast(self.words_len, tf.int32)
        """
        self.words:  tf.int64, shape=[None, None]   # shape = (batch size, max length of sentence in batch)
        self.words_len: tf.int32, shape=[None],     #   shape = (batch size)
        self.chars: tf.int64, shape=[None, None, None], # shape = (batch size, max length of sentence, max length of word)
        self.chars_len: tf.int64, shape=[None, None],   # shape = (batch_size, max_length of sentence)
        self.begin_span: tf.int32, shape=[None, None],  # shape = (batch_size, max number of candidate spans in one of the batch sentences)
        self.end_span: tf.int32, shape=[None, None],
        self.spans_len: tf.int64, shape=[None],     # shape = (batch size)
        self.cand_entities: tf.int64, shape=[None, None, None],  # shape = (batch size, max number of candidate spans, max number of cand entitites)
        self.cand_entities_scores: tf.float32, shape=[None, None, None],
        self.cand_entities_labels: tf.int64, shape=[None, None, None],
        # shape = (batch_size, max number of candidate spans)
        self.cand_entities_len: tf.int64, shape=[None, None],
        self.ground_truth: tf.int64, shape=[None, None],  # shape = (batch_size, max number of candidate spans)
        self.ground_truth_len: tf.int64, shape=[None],    # shape = (batch_size)
        self.begin_gm: tf.int64, shape=[None, None],  # shape = (batch_size, max number of gold mentions)
        self.end_gm = tf.placeholder(tf.int64, shape=[None, None],
        """
        with open(config.base_folder +"data/tfrecords/" + self.args.experiment_name +
                          "/word_char_maps.pickle", 'rb') as handle:
            _, id2word, _, id2char, _, _ = pickle.load(handle)
            self.nwords = len(id2word)
            self.nchars = len(id2char)

        self.loss_mask = self._sequence_mask_v13(self.cand_entities_len, tf.shape(self.cand_entities_scores)[2])

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def init_embeddings(self):
        print("\n!!!! init embeddings !!!!\n")
        # read the numpy file
        embeddings_nparray = np.load(config.base_folder +"data/tfrecords/" + self.args.experiment_name +
                                     "/embeddings_array.npy")
        self.sess.run(self.word_embedding_init, feed_dict={self.word_embeddings_placeholder: embeddings_nparray})

        entity_embeddings_nparray = util.load_ent_vecs(self.args)
        self.sess.run(self.entity_embedding_init, feed_dict={self.entity_embeddings_placeholder: entity_embeddings_nparray})

    """
        论文中word - char embedding部分，entity embedding部分
    """
    def add_embeddings_op(self):
        """Defines self.word_embeddings"""
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                    tf.constant(0.0, shape=[self.nwords, 300]),
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=False)    # word_embedding参数不进行更新

            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [self.nwords, 300])
            self.word_embedding_init = _word_embeddings.assign(self.word_embeddings_placeholder)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.words, name="word_embeddings")
            self.pure_word_embeddings = word_embeddings
            #print("word_embeddings (after lookup) ", word_embeddings)

        with tf.variable_scope("chars"):
            if self.args.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.nchars, self.args.dim_char], trainable=True)
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.chars, name="char_embeddings")

                # char_embeddings: tf.float32, shape=[None, None, None, dim_char],
                # shape = (batch size, max length of sentence, max length of word, dim_char)
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.args.dim_char])
                # (batch*sent_length, characters of word, dim_char)

                char_lengths = tf.reshape(self.chars_len, shape=[s[0] * s[1]])
                # shape = (batch_size*max_length of sentence)

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=char_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[s[0], s[1], 2 * self.args.hidden_size_char])
                #print("output after char lstm ", output)
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)  # concatenate word and char embeddings
                #print("word_embeddings with char after concatenation ", word_embeddings)
                # (batch, words, 300+2*100)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

        with tf.variable_scope("entities"):
            from preprocessing.util import load_wikiid2nnid
            self.nentities = len(load_wikiid2nnid(extension_name=self.args.entity_extension))
            _entity_embeddings = tf.Variable(
                tf.constant(0.0, shape=[self.nentities, 300]),
                name="_entity_embeddings",
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs)

            self.entity_embeddings_placeholder = tf.placeholder(tf.float32, [self.nentities, 300])
            self.entity_embedding_init = _entity_embeddings.assign(self.entity_embeddings_placeholder)

            self.entity_embeddings = tf.nn.embedding_lookup(_entity_embeddings, self.cand_entities,
                                                       name="entity_embeddings")
            self.pure_entity_embeddings = self.entity_embeddings
            if self.args.ent_vecs_regularization.startswith("l2"):  # 'l2' or 'l2dropout'
                self.entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, dim=3)
                # not necessary since i do normalization in the entity embed creation as well, just for safety
            if self.args.ent_vecs_regularization == "dropout" or \
                            self.args.ent_vecs_regularization == "l2dropout":
                self.entity_embeddings = tf.nn.dropout(self.entity_embeddings, self.dropout)
            #print("entity_embeddings = ", self.entity_embeddings)

    def add_context_emb_op(self):
        """this method creates the bidirectional LSTM layer (takes input the v_k vectors and outputs the
        context-aware word embeddings x_k)"""
        with tf.variable_scope("context-bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.words_len, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            self.context_emb = tf.nn.dropout(output, self.dropout)
            #print("context_emb = ", self.context_emb)  # [batch, words, 300]

    # mention representation = context_emb + soft-head attention [x_q, x_r, att]
    def add_span_emb_op(self):
        mention_emb_list = []
        # span embedding based on boundaries (start, end) and head mechanism. but do that on top of contextual bilistm
        # output or on top of original word+char embeddings. this flag determines that. The parer reports results when
        # using the contextual lstm emb as it achieves better score. Used for ablation studies.
        boundaries_input_vecs = self.word_embeddings if self.args.span_boundaries_from_wordemb else self.context_emb

        # the span embedding is modeled by g^m = [x_q; x_r; \hat(x)^m]  (formula (2) of paper)
        # "boundaries" mean use x_q and x_r.   "head" means use also the head mechanism \hat(x)^m (formula (3))
        # soft head attention
        if self.args.span_emb.find("boundaries") != -1:
            # shape (batch, num_of_cand_spans, emb)
            mention_start_emb = tf.gather_nd(boundaries_input_vecs, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span)[0]), 1), [1, tf.shape(self.begin_span)[1]]),
                 self.begin_span], 2))  # extracts the x_q embedding for each candidate span
            # the tile command creates a 2d tensor with the batch information. first lines contains only zeros, second
            # line ones etc...  because the begin_span tensor has the information which word inside this sentence is the
            # beginning of the candidate span.
            mention_emb_list.append(mention_start_emb)

            mention_end_emb = tf.gather_nd(boundaries_input_vecs, tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span)[0]), 1), [1, tf.shape(self.begin_span)[1]]),
                 tf.nn.relu(self.end_span-1)], 2))   # -1 because the end of span in exclusive  [start, end)
            # relu so that the 0 doesn't become -1 of course no valid candidate span end index is zero since [0,0) is empty
            mention_emb_list.append(mention_end_emb)
            #print("mention_start_emb = ", mention_start_emb)
            #print("mention_end_emb = ", mention_end_emb)

        mention_width = self.end_span - self.begin_span  # [batch, num_mentions]     the width of each candidate span

        # soft-head attention
        if self.args.span_emb.find("head") != -1:   # here the attention is computed
            # here the \hat(x)^m is computed (formula (2) and (3))
            self.max_mention_width = tf.minimum(self.args.max_mention_width,
                                                tf.reduce_max(self.end_span - self.begin_span))
            mention_indices = tf.range(self.max_mention_width) + \
                              tf.expand_dims(self.begin_span, 2)  # [batch, num_mentions, max_mention_width]
            mention_indices = tf.minimum(tf.shape(self.word_embeddings)[1] - 1,
                                         mention_indices)  # [batch, num_mentions, max_mention_width]
            #print("mention_indices = ", mention_indices)
            batch_index = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(mention_indices)[0]), 1), 2),
                                  [1, tf.shape(mention_indices)[1], tf.shape(mention_indices)[2]])
            mention_indices = tf.stack([batch_index, mention_indices], 3)
            # [batch, num_mentions, max_mention_width, [row,col] ]    4d tensor

            # for the boundaries we had the option to take them either from x_k (output of bilstm) or from v_k
            # the head is derived either from the same option as boundaries or from the v_k.
            head_input_vecs = boundaries_input_vecs if self.args.model_heads_from_bilstm else self.word_embeddings
            mention_text_emb = tf.gather_nd(head_input_vecs, mention_indices)
            # [batch, num_mentions, max_mention_width, 500 ]    4d tensor
            #print("mention_text_emb = ", mention_text_emb)

            with tf.variable_scope("head_scores"):
                # from [batch, max_sent_len, 300] to [batch, max_sent_len, 1]
                self.head_scores = util.projection(boundaries_input_vecs, 1)
            # [batch, num_mentions, max_mention_width, 1]
            mention_head_scores = tf.gather_nd(self.head_scores, mention_indices)
            # print("mention_head_scores = ", mention_head_scores)

            # depending on tensorflow version we do the same with different operations (since each candidate span is not
            # of the same length we mask out the invalid indices created above (mention_indices)).
            temp_mask = self._sequence_mask_v13(mention_width, self.max_mention_width)
            # still code for masking invalid indices for the head computation
            mention_mask = tf.expand_dims(temp_mask, 3)  # [batch, num_mentions, max_mention_width, 1]
            mention_mask = tf.minimum(1.0, tf.maximum(self.args.zero, mention_mask))  # 1e-3
            # formula (3) computation
            mention_attention = tf.nn.softmax(mention_head_scores + tf.log(mention_mask),
                                              dim=2)  # [batch, num_mentions, max_mention_width, 1]
            mention_head_emb = tf.reduce_sum(mention_attention * mention_text_emb, 2)  # [batch, num_mentions, emb]
            #print("mention_head_emb = ", mention_head_emb)
            mention_emb_list.append(mention_head_emb)

        self.span_emb = tf.concat(mention_emb_list, 2) # [batch, num_mentions, emb i.e. 1700] formula (2) concatenation
        #print("span_emb = ", self.span_emb)

    # FFNN_1 + <mention, entity>
    def add_lstm_score_op(self):
        with tf.variable_scope("span_emb_ffnn"):
            # [batch, num_mentions, 300]
            # the span embedding can have different size depending on the chosen hyperparameters. We project it to 300
            # dims to match the entity embeddings  (formula 4)
            if self.args.span_emb_ffnn[0] == 0:
                span_emb_projected = util.projection(self.span_emb, 300)    # F4
            else:
                hidden_layers, hidden_size = self.args.span_emb_ffnn[0], self.args.span_emb_ffnn[1]
                span_emb_projected = util.ffnn(self.span_emb, hidden_layers, hidden_size, 300,
                                               self.dropout if self.args.ffnn_dropout else None)
                #print("span_emb_projected = ", span_emb_projected)
        # formula (6) <x^m, y_j>   computation. this is the lstm score
        scores = tf.matmul(tf.expand_dims(span_emb_projected, 2), self.entity_embeddings, transpose_b=True)
        #print("scores = ", scores)
        self.similarity_scores = tf.squeeze(scores, axis=2)  # [batch, num_mentions, 1, 30]
        #print("scores = ", self.similarity_scores)   # [batch, num_mentions, 30]

    # long range attention
    def add_local_attention_op(self):
        """
        添加局部注意力机制操作，通过局部上下文更新实体嵌入的注意力分数。
        """
        # 根据参数选择实体嵌入，决定是否使用无正则化的嵌入
        attention_entity_emb = self.pure_entity_embeddings if self.args.attention_ent_vecs_no_regularization else self.entity_embeddings
        
        # 定义注意力机制的变量范围
        with tf.variable_scope("attention"):
            K = self.args.attention_K  # 注意力窗口大小
            
            # 计算左侧和右侧的掩码，确保注意力只在窗口范围内
            left_mask = self._sequence_mask_v13(self.begin_span, K)   # 左侧单词数量
            right_mask = self._sequence_mask_v13(tf.expand_dims(self.words_len, 1) - self.end_span, K)   # 右侧单词数量
            ctxt_mask = tf.concat([left_mask, right_mask], 2)  # [batch, num_of_spans, 2*K]
            ctxt_mask = tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, ctxt_mask)))
            
            # 计算左右上下文索引，确保在窗口范围内的单词
            leftctxt_indices = tf.maximum(0, tf.range(-1, -K - 1, -1) + tf.expand_dims(self.begin_span, 2))  # [batch, num_mentions, K]
            rightctxt_indices = tf.minimum(tf.shape(self.pure_word_embeddings)[1] - 1, tf.range(K) + tf.expand_dims(self.end_span, 2))  # [batch, num_mentions, K]
            ctxt_indices = tf.concat([leftctxt_indices, rightctxt_indices], 2)  # [batch, num_mentions, 2*K]
            
            # 构建batch索引，用于gather_nd操作
            batch_index = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(ctxt_indices)[0]), 1), 2), [1, tf.shape(ctxt_indices)[1], tf.shape(ctxt_indices)[2]])
            ctxt_indices = tf.stack([batch_index, ctxt_indices], 3)
            
            # 选择用于注意力计算的单词嵌入，根据参数决定是否使用LSTM输出
            att_x_w = self.pure_word_embeddings
            if self.args.attention_on_lstm and self.args.nn_components.find("lstm") != -1:
                att_x_w = util.projection(self.context_emb, 300)  # 如果context_emb的最后一个维度不等于300，则进行投影
                
            # 通过gather_nd操作获取上下文单词嵌入
            ctxt_word_emb = tf.gather_nd(att_x_w, ctxt_indices)  # [batch, num_of_spans, 2K, emb_size]
            
            # 计算单词与实体嵌入之间的注意力分数
            temp = attention_entity_emb
            if self.args.attention_use_AB:
                att_A = tf.get_variable("att_A", [300])
                temp = att_A * attention_entity_emb
            scores = tf.matmul(ctxt_word_emb, temp, transpose_b=True)  # [batch, num_of_spans, 2K]
            scores = tf.reduce_max(scores, reduction_indices=[-1])  # [batch, num_of_spans]
            scores = scores + ctxt_mask  # 更新分数以考虑掩码
            
            # 获取每个span的前R个最高分数
            top_values, _ = tf.nn.top_k(scores, self.args.attention_R)  # [batch, num_of_spans, R]
            R_value = top_values[:, :, -1]  # [batch, num_of_spans]
            R_value = tf.maximum(self.args.zero, R_value)
            
            # 构建阈值并更新分数
            threshold = tf.tile(tf.expand_dims(R_value, 2), [1, 1, 2 * K])  # [batch, num_of_spans, 2K]
            scores = scores - tf.to_float(((scores - threshold) < 0)) * 50  # [batch, num_of_spans, 2K]
            scores = tf.nn.softmax(scores, dim=2)  # [batch, num_of_spans, 2K]
            
            # 计算加权上下文表示
            x_c = tf.reduce_sum(tf.expand_dims(scores, 3) * ctxt_word_emb, 2)  # [batch, num_of_spans, emb_size]
            if self.args.attention_use_AB:
                att_B = tf.get_variable("att_B", [300])
                x_c = att_B * x_c
            x_c = tf.expand_dims(x_c, 3)  # [batch, num_of_spans, emb_size, 1]
            
            # 计算实体和上下文之间的注意力分数
            x_e__x_c = tf.matmul(attention_entity_emb, x_c)  # [batch, num_of_spans, 30, 1]
            x_e__x_c = tf.squeeze(x_e__x_c, axis=3)  # [batch, num_of_spans, 30]
            self.attention_scores = x_e__x_c
    # final local score
    def add_cand_ent_scores_op(self):
        self.log_cand_entities_scores = tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, self.cand_entities_scores)))
        stack_values = []
        # 1. <x, y> 2. attention 3. log
        if self.args.nn_components.find("lstm") != -1:
            stack_values.append(self.similarity_scores)
        if self.args.nn_components.find("pem") != -1:
            stack_values.append(self.log_cand_entities_scores)
        if self.args.nn_components.find("attention") != -1:
            stack_values.append(self.attention_scores)

        scalar_predictors = tf.stack(stack_values, 3)
        #print("scalar_predictors = ", scalar_predictors)   # [batch, num_mentions, 30, 3]

        # FFNN_2
        with tf.variable_scope("similarity_and_prior_ffnn"):
            if self.args.final_score_ffnn[0] == 0:
                self.final_scores = util.projection(scalar_predictors, 1)  # [batch, num_mentions, 30, 1]
            else:
                hidden_layers, hidden_size = self.args.final_score_ffnn[0], self.args.final_score_ffnn[1]
                self.final_scores = util.ffnn(scalar_predictors, hidden_layers, hidden_size, 1,
                                              self.dropout if self.args.ffnn_dropout else None)
            self.final_scores = tf.squeeze(self.final_scores, axis=3)  # squeeze to [batch, num_mentions, 30]
            #print("final_scores = ", self.final_scores)
    
    def add_cand_ent_scores_op_with_ensemble(self):
        self.log_cand_entities_scores = tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, self.cand_entities_scores)))
        stack_values = []
        if self.args.nn_components.find("lstm") != -1:
            stack_values.append(self.similarity_scores)
        if self.args.nn_components.find("pem") != -1:
            stack_values.append(self.log_cand_entities_scores)
        if self.args.nn_components.find("attention") != -1:
            stack_values.append(self.attention_scores)
        
        scalar_predictors = tf.stack(stack_values, 3)

        # 这个地方需要加入：从此开始是ED阶段，最优化损失函数来获取mention和真正的entity对应的评分
        # boosting模块：
        '''
            1. 将batch内的scalar向量加权
            2. 在这里执行T次提升，使用adaboost回归模型，弱学习器采用ffnn
            3. 每次迭代后的ffnn输出保存起来，每个加权a_t（adaboost根据查错率计算出的权值）然后加权
            4. 最终输出的是加权平均分数，以此作为相似度
        '''
        t = 3
        score_values = []
        alphas = []
        alpha_t = tf.zeros([scalar_predictors.shape[0]])    # 每个样本计算都会有一个alpha
        weights = tf.fill(scalar_predictors.shape[0], tf.cast(1 / scalar_predictors.shape[0]))   # 一个batch内的每个样本
        for i in range(t):
            with tf.variable_scope("adaboost_ffnn_%d" % i):
                hidden_layers, hidden_size = self.args.final_score_ffnn[0], self.args.final_score_ffnn[1]
                scoret = util.ffnn(scalar_predictors, hidden_layers, hidden_size, 1, self.args.dropout, model=self)
                score_values.append(scoret)
                # bias计算：需要以cand_entities_label进行计算
                loss1_weighted = weights * self.cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores)
                loss2_weighted = weights * (1 - self.cand_entities_labels) * tf.nn.relu(self.final_scores)
                # aplpha计算
                epsilon = 1e-10
                alpha_t = 0.5 * tf.math.log((tf.reduce_sum(loss1_weighted) + epsilon) / tf.reduce_sum(loss2_weighted))
                # 更新分布
                weights *= tf.exp(alpha_t * tf.abs(self.cand_entities_labels - scoret))
                weights /= tf.reduce_sum(weights)
                alphas.append(alpha_t)
        final_scores = tf.zeros([scalar_predictors.shape[0]], dtype=tf.float32) 
        for i in range(len(alphas)):
            final_scores += alphas[i] * score_values[i]
        self.final_scores = final_scores        
        
    # 添加全局投票操作
    def add_global_voting_op(self):
        with tf.variable_scope("global_voting"):
            # 根据损失掩码和阈值调整最终分数
            self.final_scores_before_global = - (1 - self.loss_mask) * 50 + self.final_scores
            gmask = tf.to_float(((self.final_scores_before_global - self.args.global_thr) >= 0))  # [b,s,30]

            # 应用掩码到实体嵌入
            masked_entity_emb = self.pure_entity_embeddings * tf.expand_dims(gmask, axis=3)  # [b,s,30,300] * [b,s,30,1]
            batch_size = tf.shape(masked_entity_emb)[0]
            # 计算所有选民的嵌入之和
            all_voters_emb = tf.reduce_sum(tf.reshape(masked_entity_emb, [batch_size, -1, 300]), axis=1,
                                           keep_dims=True)  # [b, 1, 300]
            # 计算每个跨度的选民嵌入之和
            span_voters_emb = tf.reduce_sum(masked_entity_emb, axis=2)  # [batch, num_of_spans, 300]
            # 计算有效选民嵌入
            valid_voters_emb = all_voters_emb - span_voters_emb
            valid_voters_emb = tf.nn.l2_normalize(valid_voters_emb, dim=2)

            # 计算全局投票分数
            self.global_voting_scores = tf.squeeze(tf.matmul(self.pure_entity_embeddings, tf.expand_dims(valid_voters_emb, axis=3)), axis=3)
            # [b,s,30,300] matmul [b,s,300,1] --> [b,s,30,1]-->[b,s,30]

            # 将最终分数和全局投票分数组合
            scalar_predictors = tf.stack([self.final_scores_before_global, self.global_voting_scores], 3)
            #print("scalar_predictors = ", scalar_predictors)   #[b, s, 30, 2]
            with tf.variable_scope("psi_and_global_ffnn"):
                # 使用全连接网络调整分数
                if self.args.global_score_ffnn[0] == 0:
                    self.final_scores = util.projection(scalar_predictors, 1)
                else:
                    hidden_layers, hidden_size = self.args.global_score_ffnn[0], self.args.global_score_ffnn[1]
                    self.final_scores = util.ffnn(scalar_predictors, hidden_layers, hidden_size, 1,
                                                  self.dropout if self.args.ffnn_dropout else None)
                # 挤压最终分数
                self.final_scores = tf.squeeze(self.final_scores, axis=3)
                #print("final_scores = ", self.final_scores)

    def add_loss_op(self):
        cand_entities_labels = tf.cast(self.cand_entities_labels, tf.float32)
        loss1 = cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores)
        loss2 = (1 - cand_entities_labels) * tf.nn.relu(self.final_scores)
        self.loss = loss1 + loss2
        if self.args.nn_components.find("global") != -1 and not self.args.global_one_loss:
            loss3 = cand_entities_labels * tf.nn.relu(self.args.gamma_thr - self.final_scores_before_global)
            loss4 = (1 - cand_entities_labels) * tf.nn.relu(self.final_scores_before_global)
            self.loss = loss1 + loss2 + loss3 + loss4
        #print("loss_mask = ", loss_mask)
        self.loss = self.loss_mask * self.loss
        self.loss = tf.reduce_sum(self.loss)
        # for tensorboard
        #tf.summary.scalar("loss", self.loss)
    # 集成学习loss计算模块
    def add_loss_op_adaboost(self):
        
        pass

    def build(self):
        """构建整个神经网络模型。
        
        该函数负责初始化模型的各个组成部分，包括占位符、嵌入层、上下文嵌入、
        LSTM得分、局部注意力机制、候选实体得分、全局投票以及训练操作。构建过程
        根据配置参数（如神经网络组件和运行模式）来决定具体包含哪些部分。

        Base Model: 只有word_embedding, ctx_emb, mention_represent, entity_emb
        """
        # 添加占位符，用于输入数据。
        self.add_placeholders()
        # 初始化嵌入操作，将词汇映射到低维空间。
        self.add_embeddings_op()
        
        # 根据配置决定是否添加LSTM相关操作。
        if self.args.nn_components.find("lstm") != -1:
            self.add_context_emb_op()  # 添加上下文嵌入操作。
            self.add_span_emb_op()     # 添加跨度嵌入操作。
            self.add_lstm_score_op()   # 添加LSTM得分操作。
        
        # 根据配置决定是否添加注意力机制相关操作。
        if self.args.nn_components.find("attention") != -1:
            self.add_local_attention_op()  # 添加局部注意力操作。
        
        # 添加候选实体得分操作。
        self.add_cand_ent_scores_op()

        if self.args.nn_components.find("adaboost") != -1:
            self.add_cand_ent_scores_op_with_ensemble()
            self.add_global_vote_op()
        
        # 根据配置决定是否添加全局投票操作。
        if self.args.nn_components.find("global") != -1:
            self.add_global_voting_op()
        
        # 在训练模式下添加损失和训练操作。
        if self.args.running_mode.startswith("train"):
            self.add_loss_op()  # 添加损失操作。
            # 添加训练操作，使用指定的学习率方法和剪裁策略。
            self.add_train_op(self.args.lr_method, self.lr, self.loss, self.args.clip)
            # 合并所有summary，用于TensorBoard可视化。
            self.merged_summary_op = tf.summary.merge_all()
        
        # 根据不同的运行模式初始化或恢复会话。
        if self.args.running_mode == "train_continue":
            self.restore_session("latest")  # 恢复最新的训练会话。
        elif self.args.running_mode == "train":
            self.initialize_session()  # 初始化会话，定义self.sess并初始化变量。
            self.init_embeddings()     # 初始化嵌入层权重。

    def _sequence_mask_v13(self, mytensor, max_width):
        """mytensor is a 2d tensor"""
        if not tf.__version__.startswith("1.4"):
            temp_shape = tf.shape(mytensor)
            temp = tf.sequence_mask(tf.reshape(mytensor, [-1]), max_width, dtype=tf.float32)
            temp_mask = tf.reshape(temp, [temp_shape[0], temp_shape[1], tf.shape(temp)[-1]])
        else:
            temp_mask = tf.sequence_mask(mytensor, max_width, dtype=tf.float32)
        return temp_mask


