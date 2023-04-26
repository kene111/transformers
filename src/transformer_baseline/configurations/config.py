
class ConfigFile:
    """ Configuration Class for the Transformer Model

        model_dim: dimension size used through out the architecture.
        hidden_layer: dimension of the hidden layer for the point-wise FFN.
        num_attention_head: The number of attention heads to be used.
        vocab_size: The number of unique words present in corpus.
        num_decoder_layers: The number of decoder layers.
        num_encoder_layers: The number of encoder layers.
        max_seq_length: Maximum length of a 'tokenized sentence'.
    """
    def __init__(self, model_dim: int = 512, 
                hidden_layer: int = 2048, 
                num_attention_head: int = 8, 
                vocab_size: int = 6000, 
                num_decoder_layers: int =6, 
                num_encoder_layers: int=6,
                max_seq_length: int =12):

        self.model_dim = model_dim
        self.hidden_layer = hidden_layer
        self.num_attention_head = num_attention_head
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.multi_head_attention_reduced_dimension = int(self.model_dim/self.num_attention_head)
        self.max_seq_length = max_seq_length
