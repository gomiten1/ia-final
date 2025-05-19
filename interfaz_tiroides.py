import gradio as gr
import numpy as np
from tiroides_ui import encoder
from tiroides_ui import categorical_cols
from tiroides_ui import predecir


def pred(age, gender, smoking, hxSmoking, hxRadioThreapy, thyroidFunction, physEx, adenopathy, pathology, focality, risk, T, N, M, stage, response):
    inputs_values = [age,
                     gender,
                     smoking,
                     hxSmoking,
                     hxRadioThreapy,
                     thyroidFunction,
                     physEx,
                     adenopathy,
                     pathology,
                     focality,
                     risk,
                     T,
                     N,
                     M,
                     stage,
                     response]

    return predecir(inputs_values)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Row():
            gr.Markdown("# Descripción \nIngresa nuevos valores para predecir recurrencia.")
            inputs_values = [gr.Number(label="Age")]
            for col in categorical_cols:
                val = gr.Dropdown(label=col, choices=encoder.categories_[categorical_cols.index(col)].tolist())
                inputs_values.append(val)
            btn = gr.Button("Diagnóstico")
        with gr.Column():
            gr.Markdown("# Resultados")
            out = gr.Markdown(elem_id="resultado")
    btn.click(fn=pred, inputs=inputs_values, outputs=out)

demo.launch()
