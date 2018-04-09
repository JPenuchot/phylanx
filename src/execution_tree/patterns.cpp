//  Copyright (c) 2017-2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License}, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives.hpp>
#include <phylanx/execution_tree/compile.hpp>

#include <string>
#include <vector>

namespace phylanx { namespace execution_tree
{
    ///////////////////////////////////////////////////////////////////////////
    std::vector<std::pair<std::string, match_pattern_type>> registered_patterns;

    void register_pattern(
        std::string const& name, match_pattern_type const& pattern)
    {
        registered_patterns.push_back(std::make_pair(name, pattern));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
#define PHYLANX_MATCH_DATA(type)                                               \
    hpx::util::make_tuple(hpx::util::get<0>(primitives::type::match_data),     \
        primitives::type::match_data)                                          \
/**/
#define PHYLANX_MATCH_DATA_VERBATIM(type)                                      \
    hpx::util::make_tuple(                                                     \
        hpx::util::get<0>(primitives::type), primitives::type)                 \
/**/

        pattern_list get_all_known_patterns()
        {
            pattern_list patterns =
            {
                // variadic functions
                PHYLANX_MATCH_DATA(console_output),
                PHYLANX_MATCH_DATA(debug_output),
                PHYLANX_MATCH_DATA(string_output),
                PHYLANX_MATCH_DATA(make_list),
                PHYLANX_MATCH_DATA(range_operation),
                // binary functions
                PHYLANX_MATCH_DATA(file_read),
                PHYLANX_MATCH_DATA(file_write),
                PHYLANX_MATCH_DATA(file_read_csv),
                PHYLANX_MATCH_DATA(file_write_csv),
#if defined(PHYLANX_HAVE_HIGHFIVE)
                PHYLANX_MATCH_DATA(file_read_hdf5),
                PHYLANX_MATCH_DATA(file_write_hdf5),
#endif
                // unary functions
                PHYLANX_MATCH_DATA(enable_tracing),
                // binary operations
                PHYLANX_MATCH_DATA(store_operation),
                //
                // compiler-specific (internal) primitives
                //
                PHYLANX_MATCH_DATA(access_argument),
                PHYLANX_MATCH_DATA(function_reference),
                PHYLANX_MATCH_DATA(wrapped_function),
                PHYLANX_MATCH_DATA(define_function),
                PHYLANX_MATCH_DATA_VERBATIM(define_function::match_data_lambda),

                PHYLANX_MATCH_DATA(variable),
                PHYLANX_MATCH_DATA(wrapped_variable),
                PHYLANX_MATCH_DATA(define_variable),
                PHYLANX_MATCH_DATA_VERBATIM(define_variable::match_data_define)
            };

            std::string car_cdr_name("car_cdr");
            for (auto const& pattern : primitives::car_cdr_operation::match_data)
            {
                patterns.push_back(hpx::util::make_tuple(car_cdr_name, pattern));
            }

//            // generic functions
//            patterns.push_back(hpx::util::make_tuple(
//                "get_seed_action", primitives::get_seed_match_data));
//            patterns.push_back(hpx::util::make_tuple(
//                "set_seed_action", primitives::set_seed_match_data));

            // patterns registered from external primitive plugins
            for (auto const& pattern : registered_patterns)
            {
                patterns.push_back(
                    hpx::util::make_tuple(pattern.first, pattern.second));
            }

            return patterns;
        }

#undef PHYLANX_MATCH_DATA_VERBATIM
#undef PHYLANX_MATCH_DATA
    }

    pattern_list const& get_all_known_patterns()
    {
        static pattern_list patterns = detail::get_all_known_patterns();
        return patterns;
    }
}}
