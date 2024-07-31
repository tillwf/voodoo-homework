import tensorflow as tf

EPSILON = tf.keras.backend.epsilon()


def mean_squared_error_log(y_true, y_pred):
    y_true = tf.maximum(tf.cast(y_true, tf.float32), EPSILON)

    y_true_log = tf.math.log(y_true)
    y_pred_log = tf.math.log(y_pred)

    return tf.keras.losses.mean_squared_error(y_true_log, y_pred_log)


def weighted_mape_tf(y_true, y_pred):
    tot =tf.cast(tf.reduce_sum(y_true), tf.float32)
    tot = tf.clip_by_value(tot, clip_value_min=1, clip_value_max=10)
    wmape = tf.realdiv(
        tf.reduce_sum(
            tf.abs(
                tf.subtract(
                    tf.cast(y_true, tf.float32),
                    tf.cast(y_pred, tf.float32)
                )
            )
        ),
        tf.cast(tot, tf.float32)
    ) * 100

    return wmape
