import gradio as gr
from openai import OpenAI
import os

def load_system_prompt(filename="system_prompt.txt"):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return "You are a professional novelist. Write engaging and immersive stories."

def generate_plot_fn(
    api_base, 
    model_name, 
    system_prompt, 
    plot_seed, 
    num_chapters, 
    language
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
            timeout=60.0
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
    language
):
    if not api_base:
        api_base = "http://localhost:1234/v1"
    if not model_name:
        model_name = "google/gemma-4-26b-a4b"
    
    client = OpenAI(base_url=api_base, api_key="lm-studio")
    
    full_text = ""
    chapter_contents = []
    
    history = [{"role": "system", "content": system_prompt}]
    
    for ch in range(1, int(num_chapters) + 1):
        # Build prompt for the current chapter
        prompt = f"Write Chapter {ch} of the novel.\n"
        prompt += f"Total Chapters planned: {int(num_chapters)}\n"
        prompt += f"Target length for this chapter: {target_tokens} tokens.\n"
        prompt += f"Language: {language}\n"
        prompt += f"Detailed Plot/Outline to follow: {plot_seed}\n"
        
        if chapter_contents:
            prompt += "\nPrevious chapters summary/context:\n"
            # Provide the last chapter content to ensure context.
            prompt += f"--- Last Chapter (Chapter {ch-1}) ---\n{chapter_contents[-1][-2000:]}\n"
            
        prompt += f"\nPlease write Chapter {ch} now. (Output ONLY the story content)"

        temp_history = history + [{"role": "user", "content": prompt}]
        
        chapter_text = ""
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=temp_history,
                max_tokens=max(int(target_tokens) + 1000, 4096), # Larger buffer for cut-offs
                stream=True,
                timeout=120.0 # Long timeout for creative writing
            )
            
            current_chapter_title = f"\n\n# 제 {ch}장\n\n" if language == "Korean" else f"\n\n# 第 {ch} 章\n\n"
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
            
            chapter_contents.append(chapter_text)
            
            # Keep history manageable
            history.append({"role": "user", "content": f"Write Chapter {ch}"})
            history.append({"role": "assistant", "content": chapter_text})
            if len(history) > 10:
                history = [history[0]] + history[-9:]
            
        except Exception as e:
            # Append error message instead of replacing the entire content
            yield full_text + chapter_text + f"\n\n[Generation Stoppped/Error]: {str(e)}", None
            break

    # Save to file in output directory with sequential numbering
    output_dir = "output"
    file_path = get_next_filename(output_dir)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    yield full_text, file_path

# Gradio Interface
with gr.Blocks(title="AI Novel Generator") as demo:
    gr.Markdown("# 🖋️ AI Novel Generator (LM Studio)")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_base = gr.Textbox(label="LM Studio API Base", value="http://localhost:1234/v1")
            model_name = gr.Textbox(label="Model Name", value="google/gemma-4-26b-a4b")
            system_prompt = gr.Textbox(
                label="System Prompt", 
                value=load_system_prompt(),
                lines=3
            )
            language = gr.Radio(["Korean", "Japanese"], label="Language", value="Korean")
            
        with gr.Column(scale=1):
            plot_seed = gr.Textbox(label="Initial Idea / Seed", placeholder="Enter the main idea...", lines=3)
            num_chapters = gr.Number(label="Number of Chapters", value=5, precision=0)
            target_tokens = gr.Number(label="Target Tokens per Chapter", value=2000, precision=0)
            
    with gr.Row():
        plot_btn = gr.Button("1. Generate Plot Outline", variant="secondary")
        stop_plot_btn = gr.Button("Stop Plot", variant="stop")
        
    plot_output = gr.Textbox(label="2. Editable Plot Outline (Review and Modify)", lines=10, interactive=True)
    
    with gr.Row():
        generate_btn = gr.Button("3. Start Novel Generation", variant="primary")
        stop_btn = gr.Button("Stop Novel", variant="stop")
    
    output_text = gr.Textbox(label="4. Generated Novel Content", lines=20, interactive=False)
    download_link = gr.File(label="5. Download Full Novel (.txt)")

    # Plot click event
    plot_click = plot_btn.click(
        fn=generate_plot_fn,
        inputs=[api_base, model_name, system_prompt, plot_seed, num_chapters, language],
        outputs=[plot_output]
    )
    stop_plot_btn.click(fn=None, inputs=None, outputs=None, cancels=[plot_click])
    
    # Novel click event
    novel_click = generate_btn.click(
        fn=generate_novel,
        inputs=[api_base, model_name, system_prompt, plot_output, num_chapters, target_tokens, language],
        outputs=[output_text, download_link]
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[novel_click])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, theme=gr.themes.Soft())
