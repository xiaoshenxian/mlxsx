# -*- coding: utf-8 -*-

import os
import sys
import time
import tensorflow as tf

def export(sess, export_path_base, model_version, inputs_tensor_info_list, outputs_tensor_info_list, method_name_list, signature_name_list, default_signature_idx=0):
    export_path=os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(model_version)))
    sys.stderr.write('Exporting model to {}\n'.format(export_path))

    start_time=time.time()
    builder=tf.saved_model.builder.SavedModelBuilder(export_path)

    signature_def_map={}
    default_signature_idx=min(max(default_signature_idx, 0), min(len(inputs_tensor_info_list), len(outputs_tensor_info_list), len(method_name_list), len(signature_name_list)))
    for idx, (the_input_map, the_output_map, method_name, signature_name) in enumerate(zip(inputs_tensor_info_list, outputs_tensor_info_list, method_name_list, signature_name_list)):
        inputs={k:tf.saved_model.utils.build_tensor_info(v) for k, v in the_input_map.items()}
        outputs={k:tf.saved_model.utils.build_tensor_info(v) for k, v in the_output_map.items()}
        signature=(tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs, method_name=method_name))
        signature_def_map[signature_name]=signature
        if idx==default_signature_idx and tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY not in signature_def_map:
            signature_def_map[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]=signature

    legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature_def_map, legacy_init_op=legacy_init_op)

    builder.save()
    sys.stderr.write('Model exported. ({0:.3f} sec)\n'.format(time.time()-start_time))
