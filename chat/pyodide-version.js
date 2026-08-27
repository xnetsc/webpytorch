/* The Pyodide release everything here runs on, in ONE place.
 *
 * It was written out in six: two workers, a third fallback, the download script, the build
 * doc and a comment. A version in six places is a version that will be five places after the
 * next upgrade, and the two that disagree are a runtime whose local copy and CDN fallback
 * are different builds -- which fails only for whoever has no local copy.
 *
 * Which release matters beyond the interpreter: the distribution's package list changes
 * between them. 0.25.0 carried 260 packages, 0.27.7 carries 351.
 */
self.PYODIDE_VERSION = '0.27.7';
self.PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v' + self.PYODIDE_VERSION + '/full/';
