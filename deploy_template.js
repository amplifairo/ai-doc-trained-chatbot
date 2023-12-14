(function() {
  // Create the chat popup container
  var chatPopup = document.createElement('div');
  chatPopup.id = 'chat-popup';
  chatPopup.className = 'chat-popup';

  // Create the chat header
  var chatHeader = document.createElement('div');
  chatHeader.className = 'chat-header';

  // Create the collapse/expand button
  var collapseExpandBtn = document.createElement('button');
  collapseExpandBtn.id = 'collapse-expand-btn';
  collapseExpandBtn.textContent = '-';
  collapseExpandBtn.onclick = toggleChatPopup;

  // Append the button to the header
  chatHeader.appendChild(collapseExpandBtn);

  // Create the iframe
  var chatIframe = document.createElement('iframe');
  chatIframe.id = 'chat-iframe';
  chatIframe.src = '___IFRAME_SOURCE___'; // Replace with your chat URL

  // Append the header and iframe to the popup container
  chatPopup.appendChild(chatHeader);
  chatPopup.appendChild(chatIframe);

  // Append the chat popup to the body
  document.body.appendChild(chatPopup);

  // Inject CSS styles
  var css = `
    .chat-popup {
      position: fixed;
      bottom: 10px;
      right: 10px;
      width: 300px;
      height: 400px;
      border: 1px solid #ccc;
      box-shadow: 0 0 10px rgba(0,0,0,0.5);
      z-index: 1000;
      display: none;
    }
    .chat-header {
      background: #f1f1f1;
      padding: 5px;
      text-align: right;
    }
    #chat-iframe {
      width: 100%;
      height: calc(100% - 30px);
      border: none;
    }
    #collapse-expand-btn {
      background: none;
      border: none;
      cursor: pointer;
    }
  `;

  var styleSheet = document.createElement('style');
  styleSheet.type = 'text/css';
  styleSheet.innerText = css;
  document.head.appendChild(styleSheet);

  // Function to toggle the chat popup
  function toggleChatPopup() {
    if (chatPopup.style.height === '400px') {
      chatPopup.style.height = '30px';
      collapseExpandBtn.textContent = '+';
    } else {
      chatPopup.style.height = '400px';
      collapseExpandBtn.textContent = '-';
    }
  }

  // Show the chat popup after a delay
  setTimeout(function() {
    chatPopup.style.display = 'block';
  }, 5000);

})();
