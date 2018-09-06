// Copyright (c) 2017-2018 Hartmut Kaiser
// Copyright (c) 2018 Shahrzad Shirzad
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PHYLANX_PRIMITIVES_NOT_EQUAL_OCT_07_2017_0212PM)
#define PHYLANX_PRIMITIVES_NOT_EQUAL_OCT_07_2017_0212PM

#include <phylanx/config.hpp>
#include <phylanx/plugins/booleans/comparison.hpp>

#include <string>
#include <utility>
#include <vector>

namespace phylanx { namespace execution_tree { namespace primitives
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct not_equal_op;
    }

    ///////////////////////////////////////////////////////////////////////////
    class not_equal : public comparison<detail::not_equal_op>
    {
        using base_type = comparison<detail::not_equal_op>;

    public:
        static match_pattern_type const match_data;

        not_equal() = default;

        not_equal(primitive_arguments_type&& operands,
            std::string const& name, std::string const& codename);
    };

    inline primitive create_not_equal(hpx::id_type const& locality,
        primitive_arguments_type&& operands,
        std::string const& name = "", std::string const& codename = "")
    {
        return create_primitive_component(
            locality, "__ne", std::move(operands), name, codename);
    }
}}}

#endif


