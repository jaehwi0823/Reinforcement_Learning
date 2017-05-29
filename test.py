import tensorflow as tf
filename_queue = tf.train.string_input_producer(["./test.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[1], [1], [1]]
col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
feature = tf.stack([col1, col2])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    label = sess.run(col3)
    coord.request_stop()
    coord.join(threads)
