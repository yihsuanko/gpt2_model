# gpt2_model

1. run_clm_loss.py

    - 設定block_size
    - tokenizer_kwargs 增加 model_max_length (注意 block_size 跟 model_max_length)
    - add_tokens : "[TIL]"  "[WS]"
    - output = tokenizer(examples[text_column_name], add_special_tokens =  False, max_length = 600, padding = "max_length", truncation = True)
    - 

2. model
    - 繼承 GPT2LMHeadModel
    - 修改forward，以只計算title的loss
        ```python
            mask = torch.where(input_ids == 21128, 1, 0)
            idx = (mask == torch.tensor(1)).nonzero().tolist()
            for ind in idx:
                mask[ind[0],ind[1]:] = 1
                mask[ind[0],ind[1]] = 0
            
            # use gpu
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            mask = mask.to(device)
            labels = labels * mask

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # gain the length of title part, and use it to calculate the loss
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
        ```
 
3. config
