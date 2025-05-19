import gradio as gr
import pandas as pd
from gliomas_ui import feature_names
from gliomas_ui import predecir


def pred(age, gender, race, idh1, tp53, atrx, pten, egfr, cic, muc16, pik3ca, nf1, pik3r1, fubp1, rb1, notch1, bcor, csmd3, smarca4, grin2a, idh2, fat4, pdgfra):

    race_map = {
        'white': 0,
        'black': 1,
        'african american': 1,
        'asian': 2,
        'american indian': 3,
        'alaska native': 3
    }

    gender_map = {'male': 0, 'female': 1}

    inputs = [age,
              gender,
              race,
              idh1,
              tp53,
              atrx,
              pten,
              egfr,
              cic,
              muc16,
              pik3ca,
              nf1,
              pik3r1,
              fubp1,
              rb1,
              notch1,
              bcor,
              csmd3,
              smarca4,
              grin2a,
              idh2,
              fat4,
              pdgfra]
    user_data = {}
    i = 0

    for fn_name in feature_names:
        print(fn_name)
        if fn_name == 'Age_at_diagnosis':
            user_data[fn_name] = inputs[i]
        elif fn_name == 'Gender':
            user_data[fn_name] = gender_map[inputs[i]]
        elif fn_name == 'Race':
            user_data[fn_name] = race_map[inputs[i]]
        else:
            user_data[fn_name] = 1 if inputs[i] == 'yes' else 0

        i = i+1
        user_df = pd.DataFrame([user_data])

    return predecir(user_df)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Row():
            gr.Markdown("# Datos del paciente\nIngresa los datos del paciente.")
            user_data = []
            for fn_name in feature_names:
                if fn_name == 'Age_at_diagnosis':
                    val = gr.Number(label=fn_name)
                elif fn_name == 'Gender':
                    val = gr.Dropdown(label=fn_name, choices=['male', 'female'])
                elif fn_name == 'Race':
                    val = gr.Dropdown(label=fn_name, choices=['white', 'black', 'african american', 'american', 'asian', 'american indian', 'alaska native'])
                else:
                    val = gr.Dropdown(label=f"Mutación en {fn_name}?", choices=['yes', 'no'])
                user_data.append(val)
            btn = gr.Button("Diagnóstico")
        with gr.Column():
            gr.Markdown("# Resultados")
            out = gr.Markdown(elem_id="resultado")
    btn.click(fn=pred, inputs=user_data, outputs=out)

demo.launch()
