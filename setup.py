from src.model.transformer import HyperbolicTransformer
from src.data.tokenizer import EnhancedTokenizer
from src.training.trainer import Trainer
from src.core.config.configurations import ModelConfig, TrainingConfig, TokenizerConfig
from torch.utils.data import DataLoader
from src.data.dataset import TextDataset  # Assumes you have a TextDataset class


# Create a training configuration instance
training_config = TrainingConfig(
    learning_rate=1e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    gradient_clip=1.0,
    num_epochs=5,  # Using 5 epochs as in your example
    batch_size=32,
    accumulation_steps=1,
    logging_steps=100,
    evaluation_steps=500,
    save_steps=1000,
    max_grad_norm=1.0,
    use_wandb=False,
    checkpoint_dir="checkpoints"
)

# Create a tokenizer configuration instance with a much lower vocab_size for testing
tokenizer_config = TokenizerConfig(vocab_size=50)  # or 53; using a low value for a small corpus

# Initialize components

tokenizer = EnhancedTokenizer(tokenizer_config)

# Define training texts (for example)
train_texts = ["Hello My name is Ahmed"]

# Train the tokenizer so that self.sp_model is initialized
tokenizer.train(train_texts, output_path="tokenizer_model")


model_config = ModelConfig(
    vocab_size=len(tokenizer.vocab),
    hidden_size=768,
    dim=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
    dropout=0.1,
    layer_norm_eps=1e-12,
    initializer_range=0.02,
    edge_importance_threshold  = 0.5
)

model = HyperbolicTransformer(model_config)
print(f"\nUsing device: {model.device}")


# Debug information
print("\nModel Configuration:")
print(f"Vocabulary Size: {model_config.vocab_size}")
print(f"Hidden Size: {model_config.hidden_size}")
print(f"Number of Layers: {model_config.num_hidden_layers}")

print("\nModel Parameters:")
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape}")
        total_params += param.numel()
print(f"\nTotal Trainable Parameters: {total_params:,}")

# Create a training dataset and DataLoader from your training texts
train_dataset = TextDataset(
    texts=train_texts,
    tokenizer=tokenizer,
    max_length=model_config.max_position_embeddings,
    is_training=True
)
train_dataloader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)

# Initialize the trainer with the model and the training configuration
trainer = Trainer(model, training_config)

# Train the model using the DataLoader
trainer.train(train_dataloader)

# Generate text using the model
output = model.generate("My name", max_length=50)
print(output)
