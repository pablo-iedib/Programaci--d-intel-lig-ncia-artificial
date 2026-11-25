import pandas as pd

species_names = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
numerical = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

def predict_single(penguin, scaler, dv, model):
    penguin_data = penguin.copy()
    df_num = pd.DataFrame([penguin_data])[numerical]
    scaled_vals = scaler.transform(df_num)[0]
    for i, col in enumerate(numerical):
        penguin_data[col] = float(scaled_vals[i])
    x = dv.transform([penguin_data])
    pred_class = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]

    return species_names[pred_class], float(probs[pred_class])