chrome.webRequest.onBeforeRequest.addListener(
    function(details) {
      if (details.url.includes("/media/antscan/processed/")) {
        return { cancel: true };
      }
      return { cancel: false };
    },
    { urls: ["<all_urls>"] },
    ["blocking"]
  );