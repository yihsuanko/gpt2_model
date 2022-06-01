# gpt2_model

1. run_clm.py

    - 設定block_size
    - tokenizer_kwargs 增加 model_max_length (注意 block_size 跟 model_max_length)
    - add_tokens : "[TIL]"  "[WS]"
    - output = tokenizer(examples[text_column_name], add_special_tokens =  False, max_length = 600, padding = "max_length", truncation = True)
    - 

2. model
    - 繼承 GPT2LMHeadModel
    - 修改forward，以只計算title的loss
        ```python
            # make sure title_id [TIL] = 21128
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
 
3. run_clm_config.json

    - "model_name_or_path": "ckiplab/gpt2-base-chinese",
    - "tokenizer": "bert-base-chinese",
    - "do_train": true,
    - "do_eval": true,
    - "train_file":"data/short_summary_train_ws_clm.csv",
    - "validation_file":"data/short_summary_test_ws_clm.csv",
    - "source_prefix": "summarize: ",
    - "output_dir": "model/gpt2_0601_test2",
    - "overwrite_output_dir":true,
    - "resume_from_checkpoint": false,
    - "per_device_train_batch_size":4,
    - "per_device_eval_batch_size":4,
    - "predict_with_generate": true,
    - "save_steps": 500,
    - "eval_steps": 500,
    - "logging_steps": 100,
    - "learning_rate": 1e-4,
    - "save_total_limit": 3,
    - "load_best_model_at_end" : true,
    - "evaluation_strategy": "steps",
    - "save_strategy":"steps",
    - "preprocessing_num_workers":4,
    - "max_seq_length": 600,
    - "max_train_samples": 5000,
    - "max_target_length": 96,
    - "generation_max_length": 96,
    - "fp16": false,
    - "early_stopping": true,
    - "warmup_ratio": 0.1,
    - "early_stopping_patience": 5,
    - "block_size": 600
