const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
const chatId = document.querySelector('#chat_id')?.value;
console.log(chatId);


$(document).on('submit', '#send-form', function(e) {
    e.preventDefault();
    $.ajaxSetup({
        headers: {
            'X-CSRFToken': csrftoken  // Set CSRF token in headers
        }
    });

    $.ajax({
        type: 'POST',
        url: '/MLS/Send',
        contentType: 'application/json',  // Ensure JSON format
        data: JSON.stringify({  // Convert data to JSON
            chat_id: $('#chat_id').val(),
            message: $('#message').val(),
        }),
        success: function(data) {
            console.log('Message sent successfully:', data);
            location.reload();
            $('#message').val(''); // Clear input field after success
        },
        error: function(error) {
            console.error('Error sending message:', error);
            alert('Failed to send the message. Please try again.');
        }
    });
});

let slideIndex = 0;
showSlide(slideIndex);

function currentSlide(n) {
    showSlide(slideIndex = n);
}

function showSlide(n) {
    const slides = document.getElementsByClassName("slide");
    const thumbs = document.getElementsByClassName("thumb");

    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove("active");
        thumbs[i].classList.remove("active");
    }

    slides[n].classList.add("active");
    thumbs[n].classList.add("active");
}