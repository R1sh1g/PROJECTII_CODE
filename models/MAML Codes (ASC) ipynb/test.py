from models.asc_infer import ASCInfer

asc = ASCInfer("models/ASC/outputs/maml_acsc_epoch4.pt")

print(
    asc.predict(
        "The battery lasts long but the charger is terrible.",
        "Battery"
    )
)
