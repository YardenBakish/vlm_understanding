'''
https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/configuration_smolvlm.py
https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SAMPLES_TO_TEST = [
"GOT-10k_Val_000001",
"GOT-10k_Val_000003",
"GOT-10k_Val_000006",
"GOT-10k_Val_000007",
"GOT-10k_Val_000015",
"GOT-10k_Val_000018",
"GOT-10k_Val_000027",
"GOT-10k_Val_000029",
"GOT-10k_Val_000030",
]


import matplotlib.font_manager as fm
'''
font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
for path in font_paths:
    print(path)
exit(1)

'''
from moviepy.editor import *


from transformers import AutoProcessor, AutoModelForImageTextToText
import torch




def create_text_image(text, width, font_size=50, bg_color=(0, 0, 0), text_color=(255, 255, 255), padding=10):
    """
    Create an image containing text with the specified width.
    
    Args:
        text (str): The text to display
        width (int): Width of the image
        font_size (int): Font size
        bg_color (tuple): Background color (R,G,B)
        text_color (tuple): Text color (R,G,B)
        padding (int): Padding around text
        
    Returns:
        PIL.Image: An image with the rendered text
    """
    # Use default font with specified size
    # Note: PIL's default font doesn't support custom sizes well
    # So we'll adjust other parameters to make the text more prominent
    font = ImageFont.truetype("FreeSansBold.ttf", font_size)

    
    # Create a temporary image to calculate text dimensions
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Adjust line height based on requested font size
    line_height = max(16, int(font_size * 0.75))  # Scale based on requested size
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        test_width = temp_draw.textbbox((0, 0), test_line, font=font)[2]
        
        if test_width <= (width - 2 * padding):
            current_line.append(word)
        else:
            if current_line:  # If the current line has words, append it
                lines.append(' '.join(current_line))
                current_line = [word]
            else:  # If a single word is too long, we still have to include it
                lines.append(word)
                current_line = []
    
    # Don't forget to add the last line
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate the height required for the text
    line_height = font_size + 4  # Add a bit of spacing between lines
    text_height = line_height * len(lines)
    
    # Create the actual image
    img = Image.new('RGB', (width, text_height + 2 * padding), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw each line of text
    y_position = padding
    for line in lines:
        draw.text((padding, y_position), line, font=font, fill=text_color)
        y_position += line_height
    
    return img


def create_video_with_text(video_files, text_list, output_path):
    """
    Creates a single video with multiple videos arranged in a row,
    with corresponding text beneath each video.
    
    Args:
        video_files (list): List of paths to video files
        text_list (list): List of texts corresponding to each video
        output_path (str): Path where the final video will be saved
    """
    if len(video_files) != len(text_list):
        raise ValueError("Number of videos and texts must match!")
    
    # Load all video clips
    video_clips = [VideoFileClip(file) for file in video_files]
    
    # Find the smallest height and use it for all clips to ensure consistency
    min_height = min(clip.h for clip in video_clips)
    
    # Resize all clips to have the same height (keeping aspect ratio)
    resized_clips = []
    for clip in video_clips:
        w_new = int(clip.w * (min_height / clip.h))
        resized_clips.append(clip.resize(height=min_height))
    
    # Create text clips using PIL and convert to MoviePy clips
    text_image_clips = []
    for i, text in enumerate(text_list):
        # Create a text image with the same width as the corresponding video
        text_img = create_text_image(text, width=resized_clips[i].w)
        
        # Convert PIL image to MoviePy ImageClip
        img_clip = ImageClip(np.array(text_img))
        img_clip = img_clip.set_duration(resized_clips[i].duration)
        
        text_image_clips.append(img_clip)
    
    # Create video-text pairs (each video with its text below)
    video_text_pairs = []
    for i in range(len(resized_clips)):
        pair = CompositeVideoClip(
            [
                resized_clips[i].set_position(('center', 'top')),
                text_image_clips[i].set_position(('center', resized_clips[i].h))
            ],
            size=(resized_clips[i].w, resized_clips[i].h + text_image_clips[i].h)
        )
        video_text_pairs.append(pair)
    
    # Arrange all pairs in one row
    final_width = sum(pair.w for pair in video_text_pairs)
    final_height = max(pair.h for pair in video_text_pairs)
    
    # Position each pair horizontally
    x_position = 0
    positioned_pairs = []
    for pair in video_text_pairs:
        positioned_pair = pair.set_position((x_position, 0))
        positioned_pairs.append(positioned_pair)
        x_position += pair.w
    
    # Create the final composition
    final_composition = CompositeVideoClip(
        positioned_pairs,
        size=(final_width, final_height)
    )
    
    # Set the duration to the maximum of all video durations
    max_duration = max(clip.duration for clip in video_clips)
    final_composition = final_composition.set_duration(max_duration)
    
    # Write the final video
    final_composition.write_videofile(
        output_path, 
        codec='libx264', 
        audio_codec='aac',
        fps=24
    )
    
    # Close all clips
    for clip in video_clips + [final_composition]:
        clip.close()
    # Image clips don't need explicit closing
    
    return output_path





model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    #_attn_implementation="flash_attention_2"
).to("cuda")


for sample in SAMPLES_TO_TEST:

    explainations = [None,None,None]
    videos = ["video_original.mp4", "video_blur_object.mp4","video_blur_full.mp4"]
    paths = [f"dataset/GOT10KVAL_teacher/{sample}/{videos[i]}" for i in range(3)]
    output_dir = f"samples/{sample}"
    os.makedirs(output_dir,  exist_ok=True)

    for i in range(len(explainations)):
        path = paths[i]
        print(path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": f"{path}"},
                    {"type": "text", "text": "Describe this video in detail"}
                ]
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        explaination = generated_texts[0].split("Assistant: ")[-1]
        explainations[i] = explaination

        print(explaination)
    
    
    print(explainations)
    create_video_with_text(paths, explainations, f"{output_dir}/final_vid.mp4")

