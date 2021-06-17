module.exports = {
  purge: {
    content: [
      "src/**/*.svelte",
      "src/**/*.html",
      "src/**/*.js",
      "src/**/*.ts",
    ],
    // These options are passed through directly to PurgeCSS
    options: {
      defaultExtractor: (content) => {
        const regExp = new RegExp(/[A-Za-z0-9-_:/\.]+/g);

        const matchedTokens = [];

        let match = regExp.exec(content);

        while (match) {
          if (match[0].startsWith("class:")) {
            matchedTokens.push(match[0].substring(6));
          } else {
            matchedTokens.push(match[0]);
          }

          match = regExp.exec(content);
        }

        return matchedTokens;
      },
    },
  },

  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {},
  },
  variants: {
    extend: {},
  },
  plugins: [],
}
