#!/usr/bin/env python

import methods
import sys
import os
import os.path

executable_name = 'brain_test'
executable_dir = '#bin'

# -----------


""" Get Arguments """
target = ARGUMENTS.get('target', "debug")
verbose = ARGUMENTS.get('verbose', False)


debug = target == 'debug'


env = Environment()


""" Create directories """
Execute(Mkdir('bin'))


""" Project building """

env.executable_name = executable_name
env.executable_dir = executable_dir
env.debug = debug

env.__class__.add_source_files = methods.add_source_files
env.__class__.add_library = methods.add_library
env.__class__.add_program = methods.add_program
env.__class__.disable_warnings = methods.disable_warnings

# default include path
env.Append(CPPPATH=[ '#' ])

if not verbose:
    methods.no_verbose(sys, env)

if debug:
    env.Append(CPPDEFINES=['DEBUG_ENABLED'])
    env.Append(CCFLAGS=['-ggdb'])

env.Append(LIBPATH=[executable_dir])

Export('env')

""" Script executions """
SConscript("core/SCsub")

""" Build test main """
if env.debug:
    executable_name += '.debug'

env.add_program(env.executable_dir + '/' + executable_name, ['main.cpp'])

