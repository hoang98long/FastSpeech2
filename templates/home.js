function sendText() {
    /* var item = {
        'text': document.getElementById("fname").value
    } */
    var json_item = JSON.stringify({
        'text': document.getElementById("fname").value
    });
    // text_data = document.getElementById("fname").value
    console.log(json_item);
    var URL = "http://127.0.0.1:80/tts/generate";
    $.ajax({
        type: "get",
        url: URL,
        data: json_item, 
        dataType: 'json',
        success: function(data) {
            console.log(data);
        }
    });
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
