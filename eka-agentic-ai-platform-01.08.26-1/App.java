package com.example; // Assuming a simple package structure

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class App {

    public static void main(String[] args) {
        try {
            HelloGanesha hello = new HelloGanesha();
            hello.run(); // Call the equivalent functionality of hello_ganesha.py
        } catch (Exception e) {
            System.err.println("Error executing HelloGanesha: " + e.getMessage());
            e.printStackTrace();
        }
    }
}