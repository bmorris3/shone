.. _dev:

***********************
Developer documentation
***********************

Editable install
----------------

You can install an editable version of ``shone`` on your machine by cloning
the source code, changing into the repository directory, and running::

    python -m pip install -e .


Contributor guide
-----------------

We'll briefly summarize how to contribute to ``shone`` here. The
`astropy contributor docs <https://docs.astropy.org/en/latest/index_dev.html>`_
are a much more thorough resource if you need more.

1. There are several optional dependencies that you'll need to run the tests and
   build the docs. You can get these dependencies with::

    python -m pip install -e '.[docs,test]' tox


Preparing your changes
++++++++++++++++++++++

2. Fork the ``shone`` repository, and clone your fork on your local machine.

3. Create a new branch in your local clone of your forked repository. Give it
   a name that refers to the bugs you intend to fixe, or the new features you'll implement.

4. Make your changes to the source code.

5. We need to know if your changes do what you expect them to, so add a test.

6. If you're adding a new feature, don't forget to leave comments throughout
   the source, make docstrings on new functions, and mention major new features elsewhere
   in the narrative docs.

Run tests, build docs
+++++++++++++++++++++

7. Test the changes to ensure that they work. To run the tests::

    tox -e test

   If you get a green message at the end that says ``congratulations :)``, your
   tests passed. Good work!

8. Ensure that the code style is uniform throughout the package with::

    tox -e codestyle

9. Build the documentation and check that your docstrings render correctly,
   and any narrative docs appear as you'd like::

    tox -e build_docs

   You can check the local build of the docs by opening ``docs/_build/html/index.html``
   in a web browser.

Make a pull request
+++++++++++++++++++

10. Use git to add the files you've created or changed, commit those changes, and push your branch
    to your fork of the ``shone`` repo. Then visit the upstream repo and create a pull request with
    your changes.
