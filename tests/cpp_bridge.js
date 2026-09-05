mergeInto(LibraryManager.library, {
  fft_report__deps: ['$UTF8ToString'],
  fft_report: function(success, detail) {
    document.body.dataset.status = success ? 'pass' : 'fail';
    document.querySelector('#log').textContent += '\n' + UTF8ToString(detail);
  }
});
