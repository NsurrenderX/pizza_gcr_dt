from transformers import PreTrainedModel, AutoConfig

config = AutoConfig.from_pretrained("/datahdd_8T/vla_pizza/rdt_checkpoint/170M_test_16chunk/checkpoint-100/")
print(config)