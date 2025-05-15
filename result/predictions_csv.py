import pandas as pd

from result import predictions


def prediction_csv(new_data):
    # Convert predictions to NumPy
    preds_np = predictions.cpu().numpy()

    # Save to CSV
    df_output = pd.DataFrame(preds_np, columns=["Predicted_Label"])
    df_output.to_csv("predictions.csv", index=False)

    df_input = pd.DataFrame(new_data.cpu().numpy())
    df_input["Predicted_Label"] = preds_np
    df_input.to_csv("input_with_predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
