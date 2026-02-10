from models.acd_infer import load_acd
from models.asc_infer import load_asc

acd = load_acd()
asc = load_asc()

text = "Our waitress took our order and then NEVER came back to our table."

print("ACD:", acd.predict(text))

# test ASC on a known aspect name (must match training)
print("ASC:", asc.predict(text, "staff"))
