import tensorflow as tf

def create_cell_starts(grid_size):
    cell_offsets = tf.cast(tf.range(grid_size), dtype=tf.float32)
    cell_offsets /= float(grid_size)
    cell_offsets_x = tf.tile(tf.expand_dims(cell_offsets, axis=0),
        [ grid_size, 1 ], name='cell_offsets_x')
    cell_offsets_y = tf.tile(tf.expand_dims(cell_offsets, axis=1),
        [ 1, grid_size ], name='cell_offsets_y')

    cell_starts = tf.stack([ cell_offsets_x, cell_offsets_y ], axis=2)

    return tf.expand_dims(cell_starts, axis=2, name='cell_starts')
