def score(text: str, model_bundle, threshold: float):

    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]

    # transform text
    X = vectorizer.transform([text])

    # probability of spam
    propensity = model.predict_proba(X)[0][1]

    # prediction
    prediction = bool(propensity >= threshold)

    return prediction, float(propensity)