import gradio as gr
import pandas as pd
import os
import shutil


def get_video_list(
    csv_file_obj, v_category, a_category, easy_category, hard_category, top_k
):
    if not csv_file_obj or not (
        v_category or a_category or easy_category or hard_category
    ):
        return gr.Dropdown(choices=[], interactive=True)

    df = pd.read_csv(csv_file_obj)
    # Convert category columns to int
    # Convert category columns to int
    audio_classes = [f"a_top_{i}" for i in range(1, top_k + 1)]
    video_classes = [f"v_top_{i}" for i in range(1, top_k + 1)]
    av_classes = [f"av_top_{i}" for i in range(1, top_k + 1)]

    df["a"] = df.apply(
        lambda row: row["label"] in row[audio_classes].values, axis=1
    ).astype(int)
    df["v"] = df.apply(
        lambda row: row["label"] in row[video_classes].values, axis=1
    ).astype(int)
    df["av"] = df.apply(
        lambda row: row["label"] in row[av_classes].values, axis=1
    ).astype(int)

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

    if video_ids:
        video_id = video_ids.split(" | ")[0]
        video_path = os.path.join(dataset_path, f"{video_id}.mp4")

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

        df = pd.read_csv(csv_file_obj)
        row = df[df["video_id"] == video_name]

        if not row.empty:
            tables = []
            target = row["label"].iloc[0]

            def highlight_predictions(val):
                color = "#90EE90" if val == target else "#FFFFFF"
                return f"background-color: {color}"

            target_class = f"<b>Target Class:</b> {target}"
            tables.append(target_class)

            # Audio labels table
            audio_labels_df = row[[f"a_top_{i}" for i in range(1, 11)]]
            audio_labels_df.columns = [f"Top: {i}" for i in range(1, 11)]
            audio_labels_df = pd.DataFrame(audio_labels_df).style.applymap(
                highlight_predictions,
            ).hide(axis="index")

            audio_labels_df = audio_labels_df.set_table_attributes('style="width:100%"')

            tables.append(audio_labels_df.to_html())

            # Video labels table
            video_labels_df = row[[f"v_top_{i}" for i in range(1, 11)]]
            video_labels_df.columns = [f"Top: {i}" for i in range(1, 11)]
            video_labels_df = pd.DataFrame(video_labels_df).style.applymap(
                highlight_predictions,
            ).hide(axis="index")

            video_labels_df = video_labels_df.set_table_attributes('style="width:100%"')
            tables.append(video_labels_df.to_html())

            # Multimodal labels table
            multimodal_labels_df = row[[f"av_top_{i}" for i in range(1, 11)]]
            multimodal_labels_df.columns = [f"Top: {i}" for i in range(1, 11)]
            multimodal_labels_df = pd.DataFrame(multimodal_labels_df).style.applymap(
                highlight_predictions,
            )
            multimodal_labels_df = multimodal_labels_df.set_table_attributes(
                'style="width:100%"'
            ).hide(axis="index")
            
            tables.append(multimodal_labels_df.to_html())

            return tables

    return [None, None, None]

def preview_caption(video_name, captions_df, evaluation_results_df):
    if isinstance(video_name, dict):
        video_name = video_name.get("choices", None)
    if isinstance(video_name, list):
        video_name = video_name[0]

    if video_name and captions_df:
        video_name = video_name.split(" | ")[0]

        captions_df = pd.read_csv(captions_df)
        prediction_df = pd.read_csv(evaluation_results_df)

        captions_row = captions_df[captions_df["video_id"] == video_name].dropna()
        target = prediction_df[prediction_df["video_id"] == video_name]["label"].iloc[0]

        def highlight_predictions(val):
                color = "#90EE90" if val == target else "#FFFFFF"
                return f"background-color: {color}"

        if not captions_row.empty:
            caption= captions_row["caption"].values[0]
            caption_classes_explanation = captions_row["explanation"].values[0]

            caption_classes = captions_row["classes"].values[0]
            caption_classes = pd.DataFrame(
                {f"class_{i}": [pred_class] for i, pred_class in enumerate(caption_classes.split(","))}
            ).style.applymap(
                highlight_predictions,
            )
            
            caption_classes = caption_classes.set_table_attributes(
                'style="width:100%"'
            ).hide(axis="index").to_html()

            return caption, caption_classes, caption_classes_explanation

    return [None, None, None]

def parse_results():
    import glob
    import os

    config = {
        "models": [],
        "captions": "captions/gemini/validation_captions.csv",
    }
    models = glob.glob("results/*")

    for model in models:
        config["models"].append(
            {
                "name": model.split("/")[-1],
                "test_predictions": os.path.join(model, "test_predictions.csv"),
            }
        )

    return config


with gr.Blocks() as demo:
    gr.Markdown("# Video Preview")

    results = parse_results()
    dataset_path = gr.Dropdown(
        label="Dataset Path",
        info="Select path to the dataset containing videos",
        choices=["/storage/slurm/zverev/datasets/vggsound/video/"],
        interactive=True,
        allow_custom_value=True,
    )
    captions_df = gr.File(value=results["captions"], visible=False)

    for model in results["models"]:
        model_name = model["name"]

        with gr.Tab(model_name):

            evaluation_results_df = gr.File(
                value=model["test_predictions"], visible=False
            )

            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Top-K",
                info="K used for Accuracy",
            )

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
            
            target_class = gr.HTML(label="Target Class")

            with gr.Accordion("Top-K predictions", open=False):
                with gr.Tab("Audio"):
                    audio_labels_table = gr.HTML(label="Labels")

                with gr.Tab("Video"):
                    video_labels_table = gr.HTML(label="Labels")

                with gr.Tab("Multimodal"):
                    multimodal_labels_table = gr.HTML(label="Labels")

            with gr.Accordion("Captions", open=False):
                caption = gr.Markdown("")
                caption_classes = gr.HTML(label="Labels")
                caption_classes_explanation = gr.Markdown("")

            video_output = gr.Video(label="Preview Video")

            get_videos_btn.click(
                fn=get_video_list,
                inputs=[
                    evaluation_results_df,
                    v_category,
                    a_category,
                    easy_category,
                    hard_category,
                    top_k,
                ],
                outputs=video_list,
            )

            video_list.change(
                fn=preview_video,
                inputs=[dataset_path, video_list],
                outputs=video_output,
            )
            video_list.change(
                fn=preview_labels,
                inputs=[video_list, evaluation_results_df],
                outputs=[
                    target_class,
                    audio_labels_table,
                    video_labels_table,
                    multimodal_labels_table,
                ],
            )

            video_list.change(
                fn=preview_caption,
                inputs=[video_list, captions_df, evaluation_results_df],
                outputs=[
                    caption,
                    caption_classes,
                    caption_classes_explanation,
                ],
            )

demo.launch()
