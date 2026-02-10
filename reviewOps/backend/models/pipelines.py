# reviewOps/backend/models/pipelines.py

class ReviewPipeline:
    def __init__(self, acd, asc):
        self.acd = acd
        self.asc = asc

    def run(self, review_text: str):
        acd_out = self.acd.predict(review_text)  # list of (aspect, conf)

        preds = []
        for aspect, a_conf in acd_out:
            asc_out = self.asc.predict(review_text, aspect)  # dict

            preds.append({
                "aspect": aspect,
                "aspect_confidence": float(a_conf),
                "sentiment": asc_out["sentiment"],
                "sentiment_confidence": float(asc_out["confidence"]),
                "sentiment_probs": asc_out.get("probs"),
            })

        return {
            "review": review_text,
            "predictions": preds,
        }
