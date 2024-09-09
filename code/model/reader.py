import tensorflow as tf
import argparse
import model.config as config
import pickle

def parse_sequence_example(serialized):
    """
    解析序列类型的Example。

    参数:
    serialized: 一个Tensor，包含一个序列化的Example。

    返回:
    一个元组，包含Context和Sequence特征。
    """
    # 定义序列特征。这些特征会在序列Example中被解析。
    sequence_features = {
        "words": tf.FixedLenSequenceFeature([], dtype=tf.int64),  # 单词序列，存储为int64类型的向量
        "chars": tf.VarLenFeature(tf.int64),  # 字序列，存储为int64类型的稀疏矩阵
        "chars_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),  # 字序列长度
        "begin_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),  # 开始跨度序列
        "end_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),  # 结束跨度序列
        "cand_entities": tf.VarLenFeature(tf.int64),  # 候选实体序列
        "cand_entities_scores": tf.VarLenFeature(tf.float32),  # 候选实体分数序列
        "cand_entities_labels": tf.VarLenFeature(tf.int64),  # 候选实体标签序列
        "cand_entities_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),  # 候选实体序列长度
        "ground_truth": tf.FixedLenSequenceFeature([], dtype=tf.int64)  # 真实标签序列
    }
    # 条件语句中添加额外的特征。这里假设条件总是为真，因此总是添加这些特征。
    if True:
        sequence_features["begin_gm"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
        sequence_features["end_gm"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

    # 解析序列Example。根据定义的context和sequence特征解析serialized Example。
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            "chunk_id": tf.FixedLenFeature([], dtype=tf.string),  # chunk的ID
            "words_len": tf.FixedLenFeature([], dtype=tf.int64),  # 单词序列长度
            "spans_len": tf.FixedLenFeature([], dtype=tf.int64),  # 跨度序列长度
            "ground_truth_len": tf.FixedLenFeature([], dtype=tf.int64)  # 真实标签序列长度
        },
        sequence_features=sequence_features)

    # 返回解析后的特征。将sparse tensor转换为dense，并组合成元组返回。
    return context["chunk_id"], sequence["words"], context["words_len"],\
           tf.sparse_tensor_to_dense(sequence["chars"]), sequence["chars_len"],\
           sequence["begin_span"], sequence["end_span"], context["spans_len"],\
           tf.sparse_tensor_to_dense(sequence["cand_entities"]),\
           tf.sparse_tensor_to_dense(sequence["cand_entities_scores"]),\
           tf.sparse_tensor_to_dense(sequence["cand_entities_labels"]),\
           sequence["cand_entities_len"],\
           sequence["ground_truth"], context["ground_truth_len"],\
           sequence["begin_gm"], sequence["end_gm"]
    #return context, sequence


def count_records_of_one_epoch(trainfiles):
    filename_queue = tf.train.string_input_producer(trainfiles, num_epochs=1)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    with tf.Session() as sess:
        sess.run(
            tf.variables_initializer(
                tf.global_variables() + tf.local_variables()
            )
        )

        # Start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        counter = 0
        try:
            while not coord.should_stop():
                fetch_vals = sess.run((key))
                #print(fetch_vals)
                counter += 1
        except tf.errors.OutOfRangeError:
            pass
        except KeyboardInterrupt:
            print("Training stopped by Ctrl+C.")
        finally:
            coord.request_stop()
        coord.join(threads)
    print("number of tfrecords in trainfiles = ", counter)
    return counter


def train_input_pipeline(filenames, args):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    #dataset = dataset.map(parse_sequence_example, num_parallel_calls=3)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=args.shuffle_capacity)
    dataset = dataset.padded_batch(args.batch_size, dataset.output_shapes)
    return dataset


def test_input_pipeline(filenames, args):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    dataset = dataset.padded_batch(args.batch_size, dataset.output_shapes)
    return dataset

if __name__ == "__main__":
    count_records_of_one_epoch(["/home/master_thesis_share/data/tfrecords/"
                               "wikiRLTD_perparagr_wthr_6_cthr_201/"
                               "train/wikidumpRLTD.txt"])


