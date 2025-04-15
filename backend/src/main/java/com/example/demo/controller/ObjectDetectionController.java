package com.example.demo.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class ObjectDetectionController {

    @GetMapping("/detect")
    public String detectObject(@RequestParam String imageUrl) {
        String fastApiUrl = "http://localhost:8000/detect?imageUrl=" + imageUrl;
        RestTemplate restTemplate = new RestTemplate();
        return restTemplate.getForObject(fastApiUrl, String.class);
    }
}
