$(function(){
    $('.nav-item').removeClass("active")
    $('.nav-item a[href="' + location.pathname + '"').parent().addClass('active')
});