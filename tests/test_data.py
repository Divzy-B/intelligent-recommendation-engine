from src.data.movielens import make_synthetic_ratings, prepare_data


def test_temporal_split_does_not_leak_future_items():
    ratings, items = make_synthetic_ratings(users=4, items=16, events_per_user=7)
    data = prepare_data(ratings, items)
    assert len(data["test_events"]) == 4
    for user, target in data["test_events"]:
        assert target not in data["test_histories"][user]
        assert len(data["test_histories"][user]) == 6
    assert all(len(history) >= 1 for _, history, _ in data["train_samples"])
