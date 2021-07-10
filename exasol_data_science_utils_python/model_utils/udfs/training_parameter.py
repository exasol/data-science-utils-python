class TrainingParameter:
    def __init__(self, epochs, batch_size, shuffle_buffer_size):
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.epochs = epochs