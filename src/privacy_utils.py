from opacus import PrivacyEngine

def setup_privacy_engine(model, optimizer, data_loader, epochs, max_grad_norm, target_epsilon, target_delta):
    """
    Attaches an Opacus PrivacyEngine to the model to enforce Differential Privacy.
    """
    privacy_engine = PrivacyEngine()
    
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    
    return privacy_engine, model, optimizer, data_loader

def get_privacy_metrics(privacy_engine, target_delta):
    """
    Returns the current privacy spend (epsilon).
    """
    epsilon = privacy_engine.get_epsilon(target_delta)
    return epsilon
