def find_best_matches(
    comparison_results,
    minimum_similarity=0.0
):
    """
    Find the best source sentence for every submitted sentence.

    The best match is selected using SBERT semantic similarity.

    minimum_similarity is provisional and will be calibrated
    experimentally later.
    """

    best_matches = {}

    for result in comparison_results:

        submitted_id = result["submitted_sentence_id"]

        sbert_score = result["sbert_score"]

        # Ignore candidates below the minimum similarity
        if sbert_score < minimum_similarity:
            continue

        if submitted_id not in best_matches:

            best_matches[submitted_id] = result

        else:

            current_best = best_matches[submitted_id]

            if sbert_score > current_best["sbert_score"]:

                best_matches[submitted_id] = result

    return list(best_matches.values())