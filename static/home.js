$.ajaxTransport("+binary", function (options, originalOptions, jqXHR) {
    // check for conditions and support for blob / arraybuffer response type
    if (window.FormData && ((options.dataType && (options.dataType == 'binary')) || (options.data && ((window.ArrayBuffer && options.data instanceof ArrayBuffer) || (window.Blob && options.data instanceof Blob))))) {
        return {
            // create new XMLHttpRequest
            send: function (headers, callback) {
                // setup all variables
                var xhr = new XMLHttpRequest(),
                    url = options.url,
                    type = options.type,
                    async = options.async || true,
                    // blob or arraybuffer. Default is blob
                    dataType = options.responseType || "blob",
                    data = options.data || null,
                    username = options.username || null,
                    password = options.password || null;

                xhr.addEventListener('load', function () {
                    var data = {};
                    data[options.dataType] = xhr.response;
                    // make callback and send data
                    callback(xhr.status, xhr.statusText, data, xhr.getAllResponseHeaders());
                });

                xhr.open(type, url, async, username, password);

                // setup custom headers
                for (var i in headers) {
                    xhr.setRequestHeader(i, headers[i]);
                }

                xhr.responseType = dataType;
                xhr.send(data);
            },
            abort: function () {
                jqXHR.abort();
            }
        };
    }
});
function sendText() {
    /* var item = {
        'text': document.getElementById("fname").value
    } */
    var json_item = JSON.stringify({
        'text': document.getElementById("fname").value
    });
    // text_data = document.getElementById("fname").value
    // console.log(json_item);
    var URL = "http://183.91.2.4:4097/tts/generate";
    $.ajax({
        type: "POST",
        url: URL,
        data: json_item, 
        dataType: 'binary',
        contentType: "application/json; charset=utf-8",
        responseType: 'arraybuffer',
        headers: { 'X-Requested-With': 'XMLHttpRequest' }
        success: function(response) {
            data = new Unit8Array(response);
            console.log(length(data));
            $('#audio_src'.attr('src', data))
            return data;
        }
    }).then(
        function(response) {
            data = new Unit8Array(response);
            console.log(length(data));
            $('#audio_src'.attr('src', data))
            return data;
        }

    );
    // var xmlHttp = new XMLHttpRequest();
    // xmlHttp.onreadystatechange = function() {
    //     if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
    //         callback(xmlHttp.responseText);
    // }
    // xmlHttp.open("GET", url, true); // true for asynchronous
    // xmlHttp.send();
    // (async() => {
    //     const rawResponse = await fetch('http://127.0.0.1:8080/tts/generate', {
    //         method: 'POST',
    //         headers: {
    //             'Accept': 'application/json',
    //             'Content-Type': 'application/json'
    //         },
    //         body: JSON.stringify({ text: String(document.getElementById("fname").value) })
    //     });
    //     const content = await rawResponse.json();

    //     console.log(content);
    // })();
}

function reloadPage() {
    location.reload()
}
