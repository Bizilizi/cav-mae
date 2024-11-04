import gradio as gr
import pandas as pd
import os
import shutil


def get_video_list(csv_file_obj, v_category, a_category, easy_category, hard_category):
    if not csv_file_obj or not (
        v_category or a_category or easy_category or hard_category
    ):
        return gr.Dropdown(choices=[], interactive=True)

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_obj.name)
    except Exception as e:
        return gr.Dropdown(choices=[], value=None, interactive=True)

    # Convert category columns to int
    df["a"] = df["a"].astype(int)
    df["v"] = df["v"].astype(int)
    df["av"] = df["av"].astype(int)

    # Filter based on selected categories
    mask = pd.Series([False] * len(df))

    if "V" in v_category:
        mask |= (df["a"] == 0) & (df["v"] == 1) & (df["av"] == 0)

    if "V & AV" in v_category:
        mask |= (df["a"] == 0) & (df["v"] == 1) & (df["av"] == 1)

    if "A" in a_category:
        mask |= (df["a"] == 1) & (df["v"] == 0) & (df["av"] == 0)

    if "A & AV" in a_category:
        mask |= (df["a"] == 1) & (df["v"] == 0) & (df["av"] == 1)

    if "V & A" in easy_category:
        mask |= (df["a"] == 1) & (df["v"] == 1) & (df["av"] == 0)

    if "V & A & AV" in easy_category:
        mask |= (df["a"] == 1) & (df["v"] == 1) & (df["av"] == 1)

    if "AV" in hard_category:
        mask |= (df["a"] == 0) & (df["v"] == 0) & (df["av"] == 1)

    if "AV not solvable" in hard_category:
        mask |= (df["a"] == 0) & (df["v"] == 0) & (df["av"] == 0)

    filtered_df = df[mask]

    # Get full paths to videos
    video_ids = (
        filtered_df[["video_id", "a", "v", "av"]]
        .apply(
            lambda x: f'{x["video_id"]} | a({x["a"]}) v({x["v"]}) av({x["av"]})',
            axis=1,
        )
        .tolist()
    )

    print(len(video_ids))

    return gr.Dropdown(
        choices=video_ids,
        value=video_ids[0] if video_ids else None,
        interactive=True,
    )


def preview_video(dataset_path, video_ids):
    if isinstance(video_ids, dict):
        video_ids = video_ids.get("choices", None)
    if isinstance(video_ids, list):
        video_ids = video_ids[0]

    print(video_ids)
    if video_ids:
        print(video_ids)
        video_id = video_ids.split(" | ")[0]
        video_path = os.path.join(dataset_path, f"{video_id}.mp4")
        print(video_path)

        if video_path and os.path.exists(video_path):
            tmp_path = "/tmp/gardio_av_preview.mp4"
            shutil.copy2(video_path, tmp_path)
            return tmp_path

    return None


def preview_labels(video_name, csv_file_obj):
    if isinstance(video_name, dict):
        video_name = video_name.get("choices", None)
    if isinstance(video_name, list):
        video_name = video_name[0]

    if video_name and csv_file_obj:
        video_name = video_name.split(" | ")[0]

        df = pd.read_csv(csv_file_obj.name)
        row = df[df["video_id"] == video_name]

        if not row.empty:
            tables = []
            # Create DataFrame with labels
            labels_df = pd.DataFrame(
                {
                    "Target": [row["label"].iloc[0]],
                    "Audio prediction": [row["a_label"].iloc[0]],
                    "Video prediction": [row["v_label"].iloc[0]],
                    "Multimodal prediction": [row["av_label"].iloc[0]],
                }
            )

            # Apply styling using pandas Styler
            target = row["label"].iloc[0]

            def highlight_predictions(val):
                color = "#90EE90" if val == target else "#FFB6C6"
                return f"background-color: {color}"

            styled_df = labels_df.style.applymap(
                highlight_predictions,
                subset=[
                    "Audio prediction",
                    "Video prediction",
                    "Multimodal prediction",
                ],
            ).set_properties(**{"background-color": "white"}, subset=["Target"])

            # Set table width to 100%
            styled_df = styled_df.set_table_attributes('style="width:100%"')
            tables.append(styled_df.to_html())

            # Audio labels table
            audio_labels_df = row[[f"a_top_{i}" for i in range(1, 11)]]
            audio_labels_df.columns = [f"Top: {i}" for i in range(1, 11)]
            audio_labels_df = audio_labels_df.style.set_table_attributes(
                'style="width:100%"'
            )

            tables.append(audio_labels_df.to_html())

            # Video labels table
            video_labels_df = row[[f"v_top_{i}" for i in range(1, 11)]]
            video_labels_df.columns = [f"Top: {i}" for i in range(1, 11)]
            video_labels_df = video_labels_df.style.set_table_attributes(
                'style="width:100%"'
            )
            tables.append(video_labels_df.to_html())

            # Multimodal labels table
            multimodal_labels_df = row[[f"av_top_{i}" for i in range(1, 11)]]
            multimodal_labels_df.columns = [f"Top: {i}" for i in range(1, 11)]
            multimodal_labels_df = multimodal_labels_df.style.set_table_attributes(
                'style="width:100%"'
            )
            tables.append(multimodal_labels_df.to_html())

            return tables

    return "<p>No data available</p>"


with gr.Blocks() as demo:
    gr.Markdown("# Video Preview")

    dataset_path = gr.Dropdown(
        label="Dataset Path",
        info="Select path to the dataset containing videos",
        choices=["/storage/slurm/zverev/datasets/vggsound/video/"],
        interactive=True,
        allow_custom_value=True,
    )
    csv_file = gr.File(label="CSV File", file_types=[".csv"])

    with gr.Row():
        v_category = gr.CheckboxGroup(
            choices=["V", "V & AV"],
            label="Deaf samples",
            info="Samples that are solvable only by video (V) or also by multimodal (AV)",
        )
        a_category = gr.CheckboxGroup(
            choices=["A", "A & AV"],
            label="Blind samples",
            info="Samples that are solvable only by audio (A) or also by multimodal (AV)",
        )
        easy_category = gr.CheckboxGroup(
            choices=["V & A", "V & A & AV"],
            label="Easy samples",
            info="Samples that are solvable by either audio or video or both",
        )
        hard_category = gr.CheckboxGroup(
            choices=["AV", "AV not solvable"],
            label="Hard samples",
            info="Samples that are solvable only by both audio and video or not solvable at all",
        )

    get_videos_btn = gr.Button("Get Videos")

    video_list = gr.Dropdown(label="Select Video")

    labels_table = gr.HTML(label="Labels")

    with gr.Accordion("Topk predictions", open=False):
        with gr.Tab("Audio"):
            audio_labels_table = gr.HTML(label="Labels")

        with gr.Tab("Video"):
            video_labels_table = gr.HTML(label="Labels")

        with gr.Tab("Multimodal"):
            multimodal_labels_table = gr.HTML(label="Labels")

    video_output = gr.Video(label="Preview Video")

    get_videos_btn.click(
        fn=get_video_list,
        inputs=[csv_file, v_category, a_category, easy_category, hard_category],
        outputs=video_list,
    )

    video_list.change(
        fn=preview_video, inputs=[dataset_path, video_list], outputs=video_output
    )
    video_list.change(
        fn=preview_labels,
        inputs=[video_list, csv_file],
        outputs=[
            labels_table,
            audio_labels_table,
            video_labels_table,
            multimodal_labels_table,
        ],
    )

demo.launch()
