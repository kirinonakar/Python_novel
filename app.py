import gradio as gr
from openai import OpenAI
import os
import re

def load_system_prompt(filename="system_prompt.txt"):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return "You are a professional novelist. Write engaging and immersive stories."

def save_system_prompt(content, filename="system_prompt.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content.strip())
        return "✅ System prompt saved successfully!"
    except Exception as e:
        return f"❌ Failed to save: {str(e)}"

def summarize_chapter(client, model, chapter_text, language, temperature=0.5):
    prompt = (
        f"Summarize the following chapter in 3-4 sentences in {language}.\n"
        f"Focus only on key plot events and character changes that are essential for continuity.\n\n"
        f"Chapter Content:\n{chapter_text[:4000]}" # Truncate for summary if too long
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

def split_plot_into_chapters(plot_seed, num_chapters):
    import re
    # Look for common chapter markers: Chapter 1, 1장, 제 1장, Chapter 01, etc.
    # We want to find the text between markers.
    chapter_plots = {}
    
    # Try to find all chapter-like markers and their positions
    pattern = re.compile(r'(Chapter\s*\d+|第?\s*\d+\s*[章장])', re.IGNORECASE)
    matches = list(pattern.finditer(plot_seed))
    
    if not matches:
        return None
    
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i+1 < len(matches) else len(plot_seed)
        content = plot_seed[start:end].strip()
        # Extract number from marker
        num_match = re.search(r'\d+', matches[i].group())
        if num_match:
            ch_num = int(num_match.group())
            chapter_plots[ch_num] = content
            
    return chapter_plots

def generate_plot_fn(
    api_base, 
    model_name, 
    system_prompt, 
    plot_seed, 
    num_chapters, 
    language,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1
):
    if not api_base:
        api_base = "http://localhost:1234/v1"
    if not model_name:
        model_name = "google/gemma-4-26b-a4b"
    
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    prompt = (
        f"Based on the following seed, create a detailed chapter-by-chapter plot outline for a {num_chapters}-chapter novel.\n"
        f"Language: {language}\n"
        f"Seed: {plot_seed}\n"
        f"Please include specific plot points for each of the {int(num_chapters)} chapters."
    )
    
    plot_content = ""
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            timeout=60.0,
            temperature=temperature,
            top_p=top_p
        )
        
        chunk_count = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                plot_content += chunk.choices[0].delta.content
                chunk_count += 1
                if chunk_count % 5 == 0:  # Throttling
                    yield plot_content
        
        yield plot_content  # Final yield
    except Exception as e:
        yield plot_content + f"\n\n[Generation Stoppped/Error]: {str(e)}"

def get_next_filename(directory, prefix="novel_", extension=".txt"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    max_num = 0
    for f in files:
        try:
            num_str = f[len(prefix):-len(extension)]
            num = int(num_str)
            if num > max_num:
                max_num = num
        except ValueError:
            continue
    
    next_num = max_num + 1
    return os.path.join(directory, f"{prefix}{next_num:03d}{extension}")

def generate_novel(
    api_base, 
    model_name, 
    system_prompt, 
    plot_seed, 
    num_chapters, 
    target_tokens, 
    language,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1
):
    if not api_base:
        api_base = "http://localhost:1234/v1"
    if not model_name:
        model_name = "google/gemma-4-26b-a4b"
    
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    full_text = ""
    chapter_summaries = []
    
    # Pre-parse chapters from plot if possible
    chapter_plots = split_plot_into_chapters(plot_seed, int(num_chapters))
    
    for ch in range(1, int(num_chapters) + 1):
        # Build improved prompt for the current chapter
        prompt = f"You are a professional novelist writing a novel in {language}.\n\n"
        prompt += f"[Book Information]\n"
        prompt += f"- Total Chapters: {int(num_chapters)}\n"
        prompt += f"- Current Chapter to Write: Chapter {ch}\n"
        
        # Specific plot for this chapter
        if chapter_plots and ch in chapter_plots:
            prompt += f"- Current Chapter Plot: {chapter_plots[ch]}\n\n"
        else:
            prompt += f"- Master Plot Outline:\n{plot_seed}\n\n"
        
        # Context Management
        if chapter_summaries:
            prompt += "[Story So Far (Summary)]\n"
            for i, summ in enumerate(chapter_summaries):
                prompt += f"Chapter {i+1}: {summ}\n"
            prompt += "\n"
            
            # Direct previous context (last 1000 chars)
            last_content = full_text.split("\n\n#")[-1] # Rudimentary way to get last chapter text
            prompt += f"[Directly Preceding Content (End of Chapter {ch-1})]\n"
            prompt += f"\"{last_content[-1000:]}\"\n\n"
            
        prompt += f"CRITICAL INSTRUCTION:\n"
        prompt += f"1. Write ONLY Chapter {ch}. Do not rush into future chapters.\n"
        prompt += f"2. Target length: ~{int(target_tokens)} tokens.\n"
        prompt += f"3. Output ONLY the story text. No meta-talk or phrases like 'Sure, here is chapter...'."

        try:
            chapter_text = ""
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max(int(target_tokens) + 1000, 4096),
                stream=True,
                timeout=180.0,
                temperature=temperature,
                top_p=top_p
            )
            
            if language == "Korean":
                current_chapter_title = f"\n\n# 제 {ch}장\n\n"
            elif language == "Japanese":
                current_chapter_title = f"\n\n# 第 {ch} 章\n\n"
            else:
                current_chapter_title = f"\n\n# Chapter {ch}\n\n"
            full_text += current_chapter_title
            yield full_text, None
            
            chunk_count = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chapter_text += content
                    chunk_count += 1
                    # Yield more frequently at the start, then throttle to reduce lag
                    if chunk_count < 10 or chunk_count % 10 == 0:
                        yield full_text + chapter_text, None
            
            full_text += chapter_text
            yield full_text, None # Ensure end of chapter is shown
            
            # Summarize this chapter for future context
            summary = summarize_chapter(client, model_name, chapter_text, language)
            chapter_summaries.append(summary)
            
        except Exception as e:
            # Append error message instead of replacing the entire content
            yield full_text + chapter_text + f"\n\n[Generation Stopped/Error]: {str(e)}", None
            break

    # Save to file in output directory with sequential numbering
    output_dir = "output"
    file_path = get_next_filename(output_dir)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    yield full_text, file_path

def batch_process(
    api_base, 
    model_name, 
    system_prompt, 
    plot_seed, 
    num_chapters, 
    target_tokens, 
    language, 
    batch_count,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1
):
    batch_count = int(batch_count)
    for i in range(batch_count):
        # 1. Generate Plot
        plot_content = ""
        plot_gen = generate_plot_fn(
            api_base, model_name, system_prompt, plot_seed, num_chapters, language,
            temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty
        )
        for p in plot_gen:
            plot_content = p
            yield plot_content, f"Batch {i+1}/{batch_count} - Generating plot...", None
            
        # 2. Generate Novel from that plot
        novel_gen = generate_novel(
            api_base, model_name, system_prompt, plot_content, num_chapters, target_tokens, language,
            temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty
        )
        for n_text, n_file in novel_gen:
            status_prefix = f"### [Batch {i+1}/{batch_count} In Progress]\n\n"
            yield plot_content, status_prefix + n_text, n_file

# Gradio Interface
with gr.Blocks(title="AI Novel Generator") as demo:
    gr.Markdown("# 🖋️ AI Novel Generator (LM Studio)")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_base = gr.Textbox(label="LM Studio API Base", value="http://localhost:1234/v1")
            model_name = gr.Dropdown(
                label="Model Name", 
                choices=[
                    "google/gemma-4-26b-a4b", 
                    "gemma-4-31b-it", 
                    "qwen/qwen3.5-35b-a3b", 
                    "qwen3.5-27b"
                ],
                value="google/gemma-4-26b-a4b",
                allow_custom_value=True
            )
            with gr.Row():
                system_prompt = gr.Textbox(
                    label="System Prompt", 
                    value=load_system_prompt(),
                    lines=4,
                    scale=4
                )
                save_prompt_btn = gr.Button("💾 Save", scale=1)
            
            system_status = gr.Textbox(label="Save Status", interactive=False, placeholder="Status will appear here...")
            language = gr.Radio(["Korean", "Japanese", "English"], label="Language", value="Korean")
            
        with gr.Column(scale=1):
            plot_seed = gr.Textbox(label="Initial Idea / Seed", placeholder="Enter the main idea...", lines=3)
            num_chapters = gr.Number(label="Number of Chapters", value=5, precision=0)
            target_tokens = gr.Number(label="Target Tokens per Chapter", value=2000, precision=0)
            
            with gr.Accordion("⚙️ Generation Parameters", open=False):
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=0.8)
                top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.05, value=0.95)
                repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1)

            gr.Markdown("---")
            gr.Markdown("### 📦 Batch Mode")
            with gr.Group():
                with gr.Row():
                    batch_count = gr.Number(label="Batch Count", value=1, precision=0, minimum=1)
                    batch_start_btn = gr.Button("🚀 Batch Start", variant="primary")
                    batch_stop_btn = gr.Button("⏹️ Stop", variant="stop")
            
    with gr.Row():
        plot_btn = gr.Button("1. Generate Plot Outline", variant="secondary")
        stop_plot_btn = gr.Button("Stop Plot", variant="stop")
        
    plot_output = gr.Textbox(label="2. Editable Plot Outline (Review and Modify)", lines=10, interactive=True)
    
    with gr.Row():
        generate_btn = gr.Button("3. Start Novel Generation", variant="primary")
        stop_btn = gr.Button("Stop Novel", variant="stop")
    
    output_text = gr.Textbox(label="4. Generated Novel Content", lines=20, interactive=False)
    download_link = gr.File(label="5. Download Full Novel (.txt)")

    # Save prompt event
    save_prompt_btn.click(
        fn=save_system_prompt,
        inputs=[system_prompt],
        outputs=[system_status]
    )

    # Plot click event
    plot_click = plot_btn.click(
        fn=generate_plot_fn,
        inputs=[
            api_base, model_name, system_prompt, plot_seed, num_chapters, language,
            temperature, top_p, repetition_penalty
        ],
        outputs=[plot_output]
    )
    stop_plot_btn.click(fn=None, inputs=None, outputs=None, cancels=[plot_click])
    
    # Novel click event
    novel_click = generate_btn.click(
        fn=generate_novel,
        inputs=[
            api_base, model_name, system_prompt, plot_output, num_chapters, target_tokens, language,
            temperature, top_p, repetition_penalty
        ],
        outputs=[output_text, download_link]
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[novel_click])

    # Batch click event
    batch_click = batch_start_btn.click(
        fn=batch_process,
        inputs=[
            api_base, model_name, system_prompt, plot_seed, num_chapters, target_tokens, language, batch_count,
            temperature, top_p, repetition_penalty
        ],
        outputs=[plot_output, output_text, download_link]
    )
    batch_stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[batch_click])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, theme=gr.themes.Soft())
