

def eval_sequence_vs_single(args):
    model_types = ["per_frame", "sequence", ]
    processor_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if "500M" in args.model_size  else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    prompt                = ""

    explainations_compare_dict   = {"sequence": {}, "per_frame": {}}
    inputs_dict                 = {}
    ext = "labels_w_BB.csv" if "BB" in args.mode else "labels.csv"
    for model_type in model_types:
        model_path = args.finetuned_dir
        prompt = args.prompt_finetune
        
        if "BB" in args.mode:
            processor = SmolVLMProcessor.from_pretrained(processor_path)
        else:
            processor = AutoProcessor.from_pretrained(processor_path)

        

        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
         #   config = config
            #_attn_implementation="flash_attention_2"
        ).to("cuda")
     
        for sample in SAMPLES_TO_TEST:

            explainations = [None,None,None]
            videos = ["video_original.mp4", "video_blur_object.mp4","video_blur_full.mp4"]
            paths = [f"dataset/GOT10KVAL_teacher/{sample}/{videos[i]}" for i in range(3)]

            output_dir = f'{args.eval_dir}/{sample}' 
          
            os.makedirs(output_dir,  exist_ok=True)
         

            for i in range(len(explainations)):
                path = paths[i]

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": f"{path}"},
                            {"type": "text", "text": prompt}
                        ]
                    },
                ]
                #processor.image_processor.video_sampling["max_frames"] = 1
                #print(processor)
                #print("\n\n")
              

                inputs, indices = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        return_frame_indices = True,
                        extended        = args.extended,
                        num_frames       = args.num_frames,
                        #do_resize     = False
                )



                if model_type == "per_frame":
                    BBs_per_frame = []
                    for idx in indices:
                 
                        inputs_per_frame = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        selected_index       = idx,
                        extended        = args.extended,
                        num_frames       = args.num_frames,
                        #do_resize     = False
                        )



                        if args.extend_frames:
                            inputs_per_frame.pixel_values = inputs_per_frame.pixel_values.repeat(1, 10, 1, 1, 1)

                        
                        inputs_per_frame = inputs_per_frame.to(model.device, dtype=torch.bfloat16)
                        print("REACHED")

                        generated_ids = model.generate(**inputs_per_frame, do_sample=False, max_new_tokens=1024)
                        generated_texts = processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                        )
                        explaination_per_frame = generated_texts[0].split("Assistant: ")[-1]
                        explaination_per_frame = explaination_per_frame.split(";")[0]
                        BBs_per_frame.append(explaination_per_frame)
                        
                    explaination = ";".join(BBs_per_frame)
                else:

                   
               
                
                    inputs = inputs.to(model.device, dtype=torch.bfloat16)
                    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
                    generated_texts = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    explaination = generated_texts[0].split("Assistant: ")[-1]

                explainations[i] = explaination

 
                if sample not in inputs_dict:
                    inputs_dict[sample] = []
                inputs_dict[sample].append(inputs)

           
            explainations_compare_dict[model_type][sample] = explainations

            if "BB" not in args.mode:
                create_video_with_text(paths, explainations, f"{output_dir}/final_vid_{model_type}.mp4")

        #explainations_compare.append(explainations[0])
        #paths = [paths[0],paths[0]]
    #print("\n\n")
    explainations_compare[0] = f"finetuned: {explainations_compare[0]}"
    explainations_compare[1] = f"Standard: {explainations_compare[1]}"

    #print(paths)

    create_video_with_text(paths, explainations_compare, f"{output_dir}/final_vid_compare.mp4")
    #if "BB" in args.mode:
    #    gen_comparisons_w_BB(args,explainations_compare_dict,inputs_dict, compare_mode="single_frame")

