# note

## modification
- [x] line 57: bfloat
- [x] line 182: add print warmup time, test how many times do we need. 
    converge since 2nd time warmup
- [x] separate static and dynamic VRAM

## Questions
<!-- - line 100: do we have video input? -->
- [x] line 149: Maybe we should remove min_new_token. Let's test it first.
    After removing the tag, it perform as TRT version.
- [x] Why the output append 2 pairs of number?
    Since we set `min_new_token=20`, which exceed the actual token it need. 

- [ ] line 167 No need message for input data?

### - [ ] TRT decode latency No
`model.run()`
```python
def run(self, input_text, input_image, input_audio, max_new_tokens):
    input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids, other_vision_inputs, other_audio_inputs, other_decoder_inputs = self.setup_inputs(
        input_text, input_image, input_audio)
    output_text = self.generate(pre_prompt,
                                post_prompt,
                                processed_image,
                                decoder_input_ids,
                                max_new_tokens,
                                other_vision_inputs=other_vision_inputs,
                                other_audio_inputs=other_audio_inputs,
                                other_decoder_inputs=other_decoder_inputs)
    return input_text, output_text
```

model.generate()
```python
    def generate(self,
                 pre_prompt,
                 post_prompt,
                 image,
                 decoder_input_ids,
                 max_new_tokens,
                 other_vision_inputs={},
                 other_audio_inputs={},
                 other_decoder_inputs={}):
        profiler.start("Generate")
        profiler.start("Preprocess")
        if 'qwen2_vl' in self.model_type:
            input_ids, input_lengths, ptuning_args, visual_features, mrope_args = self.preprocess(
                pre_prompt, post_prompt, image, other_vision_inputs,
                other_audio_inputs)
            mrope_params = MropeParams(
                mrope_rotary_cos_sin=mrope_args[0],
                mrope_position_deltas=mrope_args[1],
            )
        else:
            input_ids, input_lengths, ptuning_args, visual_features, model_runner_input = self.preprocess(
                pre_prompt, post_prompt, image, other_vision_inputs,
                other_audio_inputs)
        profiler.stop("Preprocess")

        # use prompt tuning to pass multimodal features
        # model.generate() expects the following params (see layers/embedding.py):
        # args[0]: prompt embedding table, [batch_size, multimodal_len, hidden_size], later flattened to [batch_size * multimodal_len, hidden_size]
        # args[1]: prompt task ids, [batch_size]. in multimodal case, arange(batch_size), i.e. in VILA batching mode 2, each image is treated separately in the batch instead of concated together (although the prompt embedding table has to be concated)
        # args[2]: prompt task vocab size, [1]. assuming all table has the same length, which in multimodal case equals to multimodal_len
        profiler.start("LLM")
        if self.decoder_llm and self.model_type != "mllama":
            end_id = self.tokenizer.eos_token_id
            if 'opt' in self.model_type and 'blip2' in self.model_type:
                # For BLIP2-OPT, model outputs a "\n" at the end.
                # we avoid it by using newline as the end token
                end_id = self.tokenizer.encode("\n",
                                               add_special_tokens=False)[0]

            if self.model_type == 'cogvlm':
                input_position_ids = self.prepare_position_ids_for_cogvlm(
                    input_ids)

            prompt_tasks = None
            prompt_table = None
            if not self.cpp_e2e:
                batch_size = len(input_ids)
                prompt_tasks = ",".join(
                    np.arange(batch_size, dtype=np.int32).astype(str))
                prompt_table = torch.stack([ptuning_args[0]])
                prompt_table = prompt_table.view(batch_size, -1,
                                                 prompt_table.shape[-1])

            output_ids = self.model.generate(
                input_ids,
                input_position_ids=input_position_ids
                if self.model_type == 'cogvlm' else None,
                mrope_params=mrope_params
                if self.model_type == 'qwen2_vl' else None,
                encoder_input_features=model_runner_input
                if self.cpp_e2e else None,
                sampling_config=None,
                prompt_table=prompt_table,
                prompt_tasks=prompt_tasks,
                max_new_tokens=max_new_tokens,
                end_id=end_id,
                pad_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.all_special_ids[0],
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                temperature=self.args.temperature,
                repetition_penalty=self.args.repetition_penalty,
                num_beams=self.args.num_beams,
                lora_uids=self.args.lora_task_uids,
                output_sequence_lengths=False,
                return_dict=False,
                mm_embedding_offloading=self.args.mm_embedding_offloading)
        elif self.model_type == "mllama":
            # When image is passed:
            # the shape of visual_features is [bs, 1, 4, 1025, hidden_size]
            # the shape of cross_attention_mask is [bs, decode_input_len, 1, 4]
            # When image is None, create dummy visual_features and cross_attention_mask
            if visual_features is None:
                visual_features = torch.zeros([
                    self.args.batch_size, 1, 4, 1,
                    self.model_config.hidden_size * self.runtime_mapping.tp_size
                ],
                                              dtype=self.model.dtype,
                                              device=self.device)
                dummy_cross_attention_mask = torch.zeros(
                    [self.args.batch_size, input_ids.shape[1], 1, 4],
                    dtype=bool,
                    device=self.device)
                skip_cross_attn_blocks = torch.ones([1],
                                                    dtype=torch.bool,
                                                    device='cpu')
            else:
                skip_cross_attn_blocks = torch.zeros([1],
                                                     dtype=torch.bool,
                                                     device='cpu')

            visual_features = visual_features.to(self.model.dtype).chunk(
                self.args.batch_size, dim=0)
            encoder_input_features = []
            cross_attention_masks = []
            encoder_output_lengths = []
            for batch_idx in range(self.args.batch_size):
                visual_feature = visual_features[batch_idx]
                num_vision_tokens = visual_feature.shape[3]
                visual_feature = visual_feature.reshape(
                    [-1, visual_feature.shape[-1]])
                encoder_max_input_length = visual_feature.shape[0]
                encoder_input_lengths = torch.IntTensor(
                    [encoder_max_input_length]).to(visual_feature.device)

                # prepare cross_attention_mask of context phase
                if 'cross_attention_mask' in other_decoder_inputs:
                    cross_attention_mask = other_decoder_inputs[
                        'cross_attention_mask'][batch_idx]
                else:
                    cross_attention_mask = dummy_cross_attention_mask[batch_idx]
                text_total_length, *_ = cross_attention_mask.shape
                cross_attention_mask = cross_attention_mask.repeat_interleave(
                    num_vision_tokens, dim=2)
                cross_attention_mask = cross_attention_mask.view(
                    text_total_length, -1)
                cross_attention_mask = cross_attention_mask.unsqueeze(1)
                cross_attention_mask = cross_attention_mask.to(
                    visual_feature.device).to(torch.bool).reshape(
                        [-1, cross_attention_mask.shape[-1]])

                # prepare cross_attention_mask for generation phase and concat them
                tmp_mask = [cross_attention_mask] + [
                    cross_attention_mask[-1:, :] for _ in range(max_new_tokens)
                ]
                cross_attention_mask = torch.concat(tmp_mask)

                encoder_input_features.append(visual_feature)
                cross_attention_masks.append(cross_attention_mask)
                encoder_output_lengths.append(encoder_input_lengths)

            outputs = self.model.generate(
                batch_input_ids=input_ids,
                encoder_input_ids=None,
                encoder_input_features=encoder_input_features,
                encoder_output_lengths=encoder_output_lengths,
                cross_attention_masks=cross_attention_masks,
                max_new_tokens=max_new_tokens,
                # max_attention_window_size=args.max_attention_window_size,
                # sink_token_length=args.sink_token_length,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                # length_penalty=args.length_penalty,
                # early_stopping=args.early_stopping,
                # beam_width_array=args.beam_width_array,
                repetition_penalty=self.args.repetition_penalty,
                # presence_penalty=args.presence_penalty,
                # frequency_penalty=args.frequency_penalty,
                # stop_words_list=stop_words_list,
                # bad_words_list=bad_words_list,
                # output_cum_log_probs=(args.output_cum_log_probs_npy != None),
                # output_log_probs=(args.output_log_probs_npy != None),
                # random_seed=args.random_seed,
                # lora_uids=args.lora_task_uids,
                # prompt_table=args.prompt_table_path,
                # prompt_tasks=args.prompt_tasks,
                # streaming=args.streaming,
                output_sequence_lengths=True,
                # no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                return_dict=True,
                # medusa_choices=args.medusa_choices,
                # return_all_generated_tokens=args.return_all_generated_tokens,
                # input_token_extra_ids=input_token_extra_ids,
                encoder_max_input_length=encoder_max_input_length,
                skip_cross_attn_blocks=skip_cross_attn_blocks,
            )
            if mpi_rank() == 0:
                output_ids = outputs["output_ids"]
        else:
            if self.model_type in ['nougat', 'pix2struct']:
                # Trim encoder input_ids to match visual features shape
                ids_shape = (self.args.batch_size, visual_features.shape[1])
                if self.model_type == 'nougat':
                    input_ids = torch.zeros(ids_shape, dtype=torch.int32)
                elif self.model_type == 'pix2struct':
                    input_ids = torch.ones(ids_shape, dtype=torch.int32)

            output_ids = self.model.generate(
                input_ids,
                decoder_input_ids,
                max_new_tokens,
                num_beams=self.args.num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                debug_mode=False,
                prompt_embedding_table=ptuning_args[0],
                prompt_tasks=ptuning_args[1],
                prompt_vocab_size=ptuning_args[2])

            # Reset input_lengths to match decoder_input_ids
            input_lengths = torch.ones(input_lengths.shape,
                                       dtype=input_lengths.dtype)
        profiler.stop("LLM")

        if mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            profiler.start("Tokenizer decode")
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True) for batch_idx in range(
                        min(self.args.batch_size, input_lengths.shape[0]))
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(self.args.num_beams)
            ] for batch_idx in range(
                min(self.args.batch_size, input_lengths.shape[0]))]
            profiler.stop("Tokenizer decode")
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

[TensorRT-LLM][INFO] Refreshed the MPI local session

```


## Idea
<!-- - Maybe not using parser. All var in config.py -->

## note
- Current TTFT and Total latency timer is splitted. (Run twice)
  - Can improve by using streamer through threading.
  - But we won't get output tokens number directly.