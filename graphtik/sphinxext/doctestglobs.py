# Copyright 2020-2020, Kostis Anagnostopoulos;
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""Patched doctest builder to expose doctest-runner's globals."""
import doctest
from doctest import DocTest, DocTestParser, DocTestRunner
from typing import Dict, List, Mapping

import sphinx
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.ext import doctest as extdoctest
from sphinx.locale import _, __
from sphinx.util import logging

log = logging.getLogger(__name__)


class ExposeGlobalsDocTestBuilder(extdoctest.DocTestBuilder):
    """Patched to expose *globals* from executed doctests (even empty ones). """

    name = "graphtik_plots"
    epilog = None
    run_empty_code = False

    def test_doc(self, docname: str, doctree: nodes.Node) -> None:
        """
        HACK: Method overridden to annotate all TestCode instances with their nodes,

        so as to store back on them the value of `:graphvar:` in the doctest-runner globals,
        after they have been executed.
        """
        groups: Dict[str, extdoctest.TestGroup] = {}
        add_to_all_groups = []
        TestRunner = extdoctest.SphinxDocTestRunner
        self.setup_runner = TestRunner(verbose=False, optionflags=self.opt)
        self.test_runner = TestRunner(verbose=False, optionflags=self.opt)
        self.cleanup_runner = TestRunner(verbose=False, optionflags=self.opt)

        self.test_runner._fakeout = self.setup_runner._fakeout  # type: ignore
        self.cleanup_runner._fakeout = self.setup_runner._fakeout  # type: ignore

        if self.config.doctest_test_doctest_blocks:

            def condition(node: nodes.Node) -> bool:
                return (
                    isinstance(node, (nodes.literal_block, nodes.comment))
                    and "testnodetype" in node
                ) or isinstance(node, nodes.doctest_block)

        else:

            def condition(node: nodes.Node) -> bool:
                return (
                    isinstance(node, (nodes.literal_block, nodes.comment))
                    and "testnodetype" in node
                )

        for node in doctree.traverse(condition):  # type: Element
            if self.skipped(node):
                continue

            source = node["test"] if "test" in node else node.astext()
            filename = self.get_filename_for_node(node, docname)
            line_number = self.get_line_number(node)
            if not source and not self.run_empty_code:
                log.warning(
                    __("no code/output in %s block"),
                    node.get("testnodetype", "doctest"),
                    location=(filename, line_number),
                )
            code = extdoctest.TestCode(
                source,
                type=node.get("testnodetype", "doctest"),
                filename=filename,
                lineno=line_number,
                options=node.get("options"),
            )
            # HACK: annotate the TestCode with the node
            # to store back plottable from doctest-runner globals.
            code.node = node

            node_groups = node.get("groups", ["default"])
            if "*" in node_groups:
                add_to_all_groups.append(code)
                continue
            for groupname in node_groups:
                if groupname not in groups:
                    groups[groupname] = extdoctest.TestGroup(groupname)
                groups[groupname].add_code(code)
        for code in add_to_all_groups:
            for group in groups.values():
                group.add_code(code)
        if self.config.doctest_global_setup:
            code = extdoctest.TestCode(
                self.config.doctest_global_setup, "testsetup", filename=None, lineno=0
            )
            for group in groups.values():
                group.add_code(code, prepend=True)
        if self.config.doctest_global_cleanup:
            code = extdoctest.TestCode(
                self.config.doctest_global_cleanup,
                "testcleanup",
                filename=None,
                lineno=0,
            )
            for group in groups.values():
                group.add_code(code)
        if not groups:
            return

        self._out("\nDocument: %s\n----------%s\n" % (docname, "-" * len(docname)))
        for group in groups.values():
            self.test_group(group)
        # Separately count results from setup code
        res_f, res_t = self.setup_runner.summarize(self._out, verbose=False)
        self.setup_failures += res_f
        self.setup_tries += res_t
        if self.test_runner.tries:
            res_f, res_t = self.test_runner.summarize(self._out, verbose=True)
            self.total_failures += res_f
            self.total_tries += res_t
        if self.cleanup_runner.tries:
            res_f, res_t = self.cleanup_runner.summarize(self._out, verbose=True)
            self.cleanup_failures += res_f
            self.cleanup_tries += res_t

    def _make_group_globals(self, group: dict):
        return {}

    def test_group(self, group: extdoctest.TestGroup) -> None:
        ns: Mapping = self._make_group_globals(group)

        def run_setup_cleanup(runner, testcodes, what):
            # type: (Any, List[TestCode], Any) -> bool
            examples = []
            for testcode in testcodes:
                example = doctest.Example(testcode.code, "", lineno=testcode.lineno)
                examples.append(example)
            if not examples:
                return True
            # simulate a doctest with the code
            sim_doctest = doctest.DocTest(
                examples,
                {},
                "%s (%s code)" % (group.name, what),
                testcodes[0].filename,
                0,
                None,
            )
            sim_doctest.globs = ns
            old_f = runner.failures
            self.type = "exec"  # the snippet may contain multiple statements
            runner.run(sim_doctest, out=self._warn_out, clear_globs=False)
            if runner.failures > old_f:
                return False
            return True

        # run the setup code
        if not run_setup_cleanup(self.setup_runner, group.setup, "setup"):
            # if setup failed, don't run the group
            return

        # run the tests
        for code in group.tests:
            py_code = code[0]
            if len(code) == 1:
                # ordinary doctests (code/output interleaved)
                try:
                    test = extdoctest.parser.get_doctest(
                        py_code.code,
                        {},
                        group.name,  # type: ignore
                        py_code.filename,
                        py_code.lineno,
                    )
                except Exception as ex:
                    log.warning(
                        __("ignoring invalid doctest code: %r\n due to: %s"),
                        py_code.code,
                        ex,
                        location=(py_code.filename, py_code.lineno),
                    )
                    continue

                # HACK: allow collecting vars even if code empty..
                if not test.examples and not self.run_empty_code:
                    continue

                for example in test.examples:
                    # apply directive's comparison options
                    new_opt = py_code.options.copy()
                    new_opt.update(example.options)
                    example.options = new_opt
                self.type = "single"  # as for ordinary doctests
            else:
                # testcode and output separate
                output = code[1] and code[1].code or ""
                options = code[1] and code[1].options or {}
                # disable <BLANKLINE> processing as it is not needed
                options[doctest.DONT_ACCEPT_BLANKLINE] = True
                # find out if we're testing an exception
                m = extdoctest.parser._EXCEPTION_RE.match(output)  # type: ignore
                if m:
                    exc_msg = m.group("msg")
                else:
                    exc_msg = None
                example = doctest.Example(
                    py_code.code,
                    output,
                    exc_msg=exc_msg,
                    lineno=py_code.lineno,
                    options=options,
                )
                test = doctest.DocTest(
                    [example], {}, group.name, py_code.filename, py_code.lineno, None
                )
                self.type = "exec"  # multiple statements again
            # DocTest.__init__ copies the globs namespace, which we don't want
            test.globs = ns
            # also don't clear the globs namespace after running the doctest
            self.test_runner.run(test, out=self._warn_out, clear_globs=False)

            ## HACK: collect plottable from doctest-runner globals.
            self._globals_updated(py_code, ns)

        # run the cleanup
        run_setup_cleanup(self.cleanup_runner, group.cleanup, "cleanup")

    def _globals_updated(self, code: extdoctest.TestCode, globs: dict):
        """Called after each test-code has executed."""
        pass
