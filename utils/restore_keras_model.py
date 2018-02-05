from __future__ import absolute_import, division, print_function
import h5py


def restore_keras_model(model, save_path, build_train=True):
    model.load_weights(save_path, by_name=True)
    with h5py.File(save_path, mode='r') as f:
        # Set optimizer weights.
        if 'optimizer_weights' in f:
            if build_train:
                # Build train function (to get weight updates).
                model._make_train_function()
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [n.decode('utf8') for n in
                                      optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]
            model.optimizer.set_weights(optimizer_weight_values)

    return model