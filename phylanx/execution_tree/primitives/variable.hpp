//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PHYLANX_PRIMITIVES_VARIABLE_SEP_05_2017_1105AM)
#define PHYLANX_PRIMITIVES_VARIABLE_SEP_05_2017_1105AM

#include <phylanx/config.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/execution_tree/primitives/base_primitive.hpp>

#include <hpx/include/components.hpp>

#include <string>
#include <vector>

namespace phylanx { namespace execution_tree { namespace primitives
{
    class variable
      : public base_primitive
      , public hpx::components::locking_hook<
          hpx::components::component_base<variable>>
    {
    public:
        variable() = default;
        PHYLANX_EXPORT variable(std::string name);

        PHYLANX_EXPORT variable(primitive_argument_type&& operand);
        PHYLANX_EXPORT variable(std::vector<primitive_argument_type>&& operands);

        PHYLANX_EXPORT variable(
            primitive_argument_type&& operand, std::string name);
        PHYLANX_EXPORT variable(
            std::vector<primitive_argument_type>&& operands, std::string name);

        PHYLANX_EXPORT primitive_argument_type eval_direct(
            std::vector<primitive_argument_type> const& params) const override;
        PHYLANX_EXPORT void store(primitive_argument_type && data) override;

    private:
        mutable primitive_argument_type data_;
        std::string name_;
        mutable bool evaluated_;
    };
}}}

#endif


