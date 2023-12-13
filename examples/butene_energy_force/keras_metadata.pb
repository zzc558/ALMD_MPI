
�root"_tf_keras_model*�{"name": "energy_gradient_model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "EnergyGradientModel", "config": {"atoms": 12, "states": 2, "invd_index": true, "angle_index": null, "dihed_index": null, "nn_size": 100, "depth": 3, "activ": "selu", "use_reg_activ": null, "use_reg_weight": null, "use_reg_bias": null, "use_dropout": false, "dropout": 0.01, "normalization_mode": 1, "energy_only": false, "precomputed_features": false, "output_as_dict": false}, "shared_object_id": 0, "is_graph_network": false, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 12, 3]}, "float32", "input_1"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 12, 3]}, "float32", "input_1"]}, "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "EnergyGradientModel", "config": {"atoms": 12, "states": 2, "invd_index": true, "angle_index": null, "dihed_index": null, "nn_size": 100, "depth": 3, "activ": "selu", "use_reg_activ": null, "use_reg_weight": null, "use_reg_bias": null, "use_dropout": false, "dropout": 0.01, "normalization_mode": 1, "energy_only": false, "precomputed_features": false, "output_as_dict": false}}}2
�root.feat_layer"_tf_keras_layer*�{"name": "feat_geo", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "FeatureGeometric", "config": {"name": "feat_geo", "trainable": true, "dtype": "float32", "invd_shape": {"class_name": "__tuple__", "items": [66, 2]}, "angle_shape": null, "dihed_shape": null}, "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 3]}}2
�root.std_layer"_tf_keras_layer*�{"name": "feat_std", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConstLayerNormalization", "config": {"name": "feat_std", "trainable": true, "dtype": "float32", "axs": -1}, "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66]}}2
�root.mlp_layer"_tf_keras_layer*�{"name": "mlp", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MLP", "config": {"name": "mlp", "trainable": true, "dtype": "float32", "dense_units": 100, "dense_depth": 3, "dense_bias": true, "dense_bias_last": true, "dense_activ": "selu", "dense_activ_last": "selu", "dense_activity_regularizer": null, "dense_kernel_regularizer": null, "dense_bias_regularizer": null, "dropout_use": false, "dropout_dropout": 0.01}, "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66]}}2
�root.energy_layer"_tf_keras_layer*�{"name": "energy", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "energy", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}2
�
root.force"_tf_keras_layer*�{"name": "force", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EmptyGradient", "config": {"name": "force", "trainable": true, "dtype": "float32", "mult_states": 2, "atoms": 12}, "shared_object_id": 8, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 3]}}2
�root.feat_layer.invd_layer"_tf_keras_layer*�{"name": "inverse_distance_indexed_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "InverseDistanceIndexed", "config": {"name": "inverse_distance_indexed_1", "trainable": true, "dtype": "float32", "invd_shape": {"class_name": "__tuple__", "items": [66, 2]}}, "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 3]}}2
�root.feat_layer.flat_layer"_tf_keras_layer*�{"name": "feat_flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "feat_flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 11}}2
� root.mlp_layer.mlp_dense_last"_tf_keras_layer*�{"name": "mlp_last", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "mlp_last", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}2
�X root.mlp_layer.mlp_dense_activ.0"_tf_keras_layer*�{"name": "mlp_dense_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "mlp_dense_0", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 66}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66]}}2
�Y root.mlp_layer.mlp_dense_activ.1"_tf_keras_layer*�{"name": "mlp_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "mlp_dense_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}2