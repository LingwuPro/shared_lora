def sample(inps: Union[Tuple[torch.Tensor, ...], torch.Tensor], size: int):
    if isinstance(inps, torch.Tensor):
        assert len(inps.shape) == 3 and inps.shape[0] == 1
        inps = inps.squeeze(0).cpu()  # (seq_length, hidden_size)
        size = min(inps.shape[0], size)
        indices = np.random.choice(inps.shape[0], size, replace=False)
        indices = torch.from_numpy(indices)
        return inps[indices]
    else:
        return tuple(sample(x, size) for x in inps)