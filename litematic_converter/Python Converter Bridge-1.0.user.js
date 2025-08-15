// ==UserScript==
// @name         Python Converter Bridge
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  try to take over the world!
// @author       You
// @match        https://abfielder.com/tools/converter/litematictoschem
// @icon         https://www.google.com/s2/favicons?sz=64&domain=abfielder.com
// @grant        GM_xmlhttpRequest
// @connect      localhost
// @run-at       document-start
// ==/UserScript==

(function() {
    'use strict';

    const PYTHON_SERVER_URL = 'http://localhost:8766/data';

    function sendSessionDataToPython() {
        const payload = {
            cookies: document.cookie,
            userAgent: navigator.userAgent
        };

        if (!payload.cookies) {
            console.error('No cookies found!');
            return;
        }

        GM_xmlhttpRequest({
            method: 'POST',
            url: PYTHON_SERVER_URL,
            headers: { 'Content-Type': 'application/json' },
            data: JSON.stringify(payload),
            onload: function(response) {
                console.log('Success! Python has the session.');
                document.documentElement.style.backgroundColor = '#dff0d8';
                setTimeout(() => window.close(), 1500);
            },
            onerror: function(response) {
                console.error('ERROR: Could not connect to Python server.', response);
                document.documentElement.style.backgroundColor = '#f2dede';
                alert('Could not connect to the Python converter server.');
            }
        });
    }

    sendSessionDataToPython();
})();