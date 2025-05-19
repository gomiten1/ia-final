import gradio as gr
from tumores_ui import feature_names
from tumores_ui import predecir


def pred(radius, texture, perimeter, area, smoothness, compactness, concavity, concavePoints, symmetry, fractalDimension):
    user_vals = [radius,
                 texture,
                 perimeter,
                 area,
                 smoothness,
                 compactness,
                 concavity,
                 concavePoints,
                 symmetry,
                 fractalDimension]

    return predecir(user_vals, feature_names)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    inputs_medidas = []
    with gr.Row():
        with gr.Row():
            gr.Markdown("# Descripción del tumor\nIngresa las medidas del tumor para clasificar.")
            for fn_name in feature_names:
                medida_inp = gr.Number(label=fn_name)
                inputs_medidas.append(medida_inp)
            btn = gr.Button("Diagnóstico")
        with gr.Column():
            gr.Markdown("# Resultados")
            out = gr.Markdown(elem_id="resultado")
    btn.click(fn=pred, inputs=inputs_medidas, outputs=out)

demo.launch()
