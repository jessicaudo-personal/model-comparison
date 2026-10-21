# Transformer Basics
The Transformer architecture was introduced in the 2019 paper "Attention Is All You Need" by Vaswani and colleagues. It replaced recurrence with attention, which allows tokens to be processed in parallel during training.

# Model Memory
A rough rule for memory is parameters multiplied by bytes per weight. At 16-bit precision each weight takes 4 bytes, so a 7-billion-parameter model needs about 28 GB for its weights alone. Quantizing to 4 bits cuts this to roughly a quarter of that size.
