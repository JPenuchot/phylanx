//  Copyright (c) 2018-2019 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/plugins/keras_support/keras_support.hpp>
#include <phylanx/plugins/plugin_factory.hpp>

#include <string>

PHYLANX_REGISTER_PLUGIN_MODULE();

PHYLANX_REGISTER_PLUGIN_FACTORY(one_hot_operation_plugin,
    phylanx::execution_tree::primitives::one_hot_operation::match_data);
PHYLANX_REGISTER_PLUGIN_FACTORY(softmax_operation_plugin,
    phylanx::execution_tree::primitives::softmax_operation::match_data);
PHYLANX_REGISTER_PLUGIN_FACTORY(softplus_operation_plugin,
    phylanx::execution_tree::primitives::softplus_operation::match_data);
