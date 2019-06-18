export const maxWordLength = 30;
export const wordVectorSize = maxWordLength * 26;

export function getWordsData(tf: any, words: string[]) {
  const wordsVectors = new Float32Array(wordVectorSize * words.length).fill(0);
  words.forEach((w, i) => {
    wordsVectors.fill(-.04, i * wordVectorSize, i * wordVectorSize + w.length * 26);
    w.toLowerCase().substr(0, maxWordLength).split('').forEach((letter, j) => {
      wordsVectors[i * wordVectorSize + j * 26 + letter.charCodeAt(0) - 97] = 1;
    });
  })
  const data = tf.tensor3d(wordsVectors, [words.length, maxWordLength, 26]);
  return data;
}

export const testWords = [
  'test', 'words', 'sglhsdzf', 'nqoewrv', 'zhangshuaige',
  'hSkqOaTZ', 'randomWords', 'pageSize', 'nfobl', 'asdfghjkl',
  'nihongo', 'karteversity', 'rhpoqe', 'jrtg', 'rthio',
  'pinyin', 'neiwai', 'python', 'javascript', 'tapqst',
  'teqats', 'waqsty', 'sfaty', 'param', 'tokyo',
  'tsunami', 'kyoto', 'faq', 'mas', 'mass',
  'applyform', 'eriguerwqhtu', 'dtehzwg', 'a', 'of',
  'form', 'qe', 'tesraper', 'qualitivation', 'staihe',
  'por', 'tesave', 'waap', 'tempt', '',
  'var', 'qutaumn', 'departare', 'tion', 'in',
  'let', 'pass', 'do', 'off', 'the',
  'flkja', 'grhu', 'sfbljk', 'vf', 'wenuota',
  'basic', 'saflingde', 'parase', 'herrow', 'dest',
  'hwqruoq', 'fdshglk', 'sfblka', 'dfbh', 'dsbflisghdfliughlfd',
  'aaaaa', 'bbbbb', 'eeeee', 'xxxxx', 'qqqqq',
  'kkbjdsav', 'dfsblfg', 'sfgfdsguifsd', 'aelkrastmastquam', 'aelkrast',
  'spass', 'ditable', 'swata', 'doaevents', 'stashment',
  'skamstablet', 'stmawavtkqm', 'patenametasd', 'paremstbayke', 'staempatsxadeque',
  'tasorments', 'statefulless', 'reimbrusment', 'tamporstham', 'questionare',
  'stlndlvnans', 'vablared', 'staramus', 'tambo', 'ham',
  'sfhdgjlkdngtrkl', 'fdlhjgkfasjkgld', 'fglhjksdfg', 'ieorytwdsa', 'adjn',
  'dfsja', 'kafas', 'parsta', 'qua', 'l',
  'thisisatest', 'iamhere', 'ilovetensorflow', 'thisisasimplesentence', 'thawakqmatquede',
  'watashihanekodesu', 'youcangetthemtogether', 'manymanywordsarejoined', 'fastclick', 'thisisaqadcgpkword',
  'zheshiyijupinyin', 'alibaba', 'alibabainc', 'alibabacorp', 'watashihachugokujindesu',
  'keras', 'samsung', 'google', 'baidu', 'tencent',
  'qa', 'cctv', 'bpms', 'pd', 'pm',
  'chrome', 'vscode', 'visualstudiocode', 'inc', 'ta',
  'st', 'tm', 'wq', 'as', 'zx',
];